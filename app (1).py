"""
Well-Log Facies Prediction — Streamlit App
--------------------------------------------
Upload a LAS file, predict facies down the depth column using the trained
Random Forest model, and view the result as a well log track.

Run locally:
    streamlit run app.py

Requires these files in the same folder (produced by the training notebook):
    best_baseline_rf.joblib   - trained Random Forest model (retrained on LAS-available curves only)
    scaler.joblib             - fitted StandardScaler (must match the model above)

NOTE ON THE MODEL:
The training dataset (SEG 2016 / Kansas Geological Survey) includes two geologic
indicator columns — NM_M and RELPOS — that are never present in a real LAS file.
The model loaded here must be the *LAS-compatible* version retrained without those
columns. See the notebook section "LAS-Compatible Model Export" for instructions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import joblib
import lasio
import io

st.set_page_config(page_title="Well-Log Facies Predictor", layout="wide")

st.title("🪨 Well-Log Facies Prediction")
st.markdown(
    "Upload a `.las` file to predict rock facies down the depth column using a "
    "Random Forest model trained on the SEG 2016 / Kansas Geological Survey "
    "facies dataset (Council Grove Field, Hugoton & Panoma gas reservoirs)."
)

# ── Facies metadata ────────────────────────────────────────────────────────────
FACIES_LABELS = {
    1: "Nonmarine sandstone",
    2: "Nonmarine coarse siltstone",
    3: "Nonmarine fine siltstone",
    4: "Marine siltstone & shale",
    5: "Mudstone",
    6: "Wackestone",
    7: "Dolomite",
    8: "Packstone-grainstone",
    9: "Bafflestone",
}
FACIES_COLORS = [
    '#F4D03F', '#F5B041', '#DC7633', '#A11D33', '#1B4F72',
    '#2E4053', '#7D6608', '#117A65', '#145A32'
]
CMAP_FACIES = mcolors.ListedColormap(FACIES_COLORS, 'indexed')

# ── Exact feature list the LAS-compatible model was trained on ─────────────────
# These are the 5 raw LAS curves + 1 ratio feature + 10 rolling features (mean & std
# over a 3-sample window for each of the 5 curves). Total: 16 features.
# ORDER MUST MATCH the column order used during scaler.fit_transform() in training.
RAW_CURVES = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']

ENGINEERED_FEATURES = (
    RAW_CURVES
    + ['GR_PHIND_ratio']
    + [f'{c}_roll_mean' for c in RAW_CURVES]
    + [f'{c}_roll_std'  for c in RAW_CURVES]
)
# Total: 5 + 1 + 5 + 5 = 16 columns — exactly what the scaler expects.


# ── Model loader (cached so it only runs once per session) ─────────────────────
@st.cache_resource
def load_model_and_scaler():
    model  = joblib.load("best_baseline_rf.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


# ── Feature engineering (must mirror the training notebook exactly) ────────────
def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to a DataFrame that already contains the 5 raw curves.
    Rolling stats are computed on the whole uploaded well (treated as one sequence).
    """
    df = df.copy()

    # 1. Lithology-porosity ratio proxy
    df['GR_PHIND_ratio'] = df['GR'] / (df['PHIND'] + 0.001)

    # 2. Rolling mean and std over a 3-sample depth window
    for curve in RAW_CURVES:
        df[f'{curve}_roll_mean'] = (
            df[curve].rolling(window=3, min_periods=1).mean()
        )
        df[f'{curve}_roll_std'] = (
            df[curve].rolling(window=3, min_periods=1).std().fillna(0)
        )

    return df


# ── PE imputation (mirrors the per-well median → global fallback in training) ──
def impute_pe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    well_median = df['PE'].median()
    if pd.isna(well_median):
        # If the whole PE column is missing, use a typical carbonate/shale value
        df['PE'] = 3.0
        st.warning(
            "PE curve is entirely missing from this LAS file. "
            "Filled with a placeholder value of 3.0 — predictions for PE-sensitive "
            "facies (Dolomite, Wackestone) may be less reliable."
        )
    else:
        df['PE'] = df['PE'].fillna(well_median)
    return df


# ── Legend helper ──────────────────────────────────────────────────────────────
def facies_legend_patches():
    return [
        mpatches.Patch(color=FACIES_COLORS[i - 1], label=f"{i} — {FACIES_LABELS[i]}")
        for i in range(1, 10)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader("Upload a LAS file", type=["las"])

if uploaded_file is not None:
    try:
        # ── Parse LAS ─────────────────────────────────────────────────────────
        las_bytes = uploaded_file.read().decode("utf-8", errors="ignore")
        las = lasio.read(io.StringIO(las_bytes))
        df = las.df().reset_index()
        depth_col = df.columns[0]
        df = df.rename(columns={depth_col: "Depth"})

        st.success(
            f"✅ Loaded **{uploaded_file.name}** — "
            f"**{len(df):,}** depth samples · "
            f"**{len(df.columns) - 1}** curves found."
        )

        with st.expander("Curves detected in this LAS file"):
            st.write(list(df.columns))

        # ── Check required curves ──────────────────────────────────────────────
        missing_curves = [c for c in RAW_CURVES if c not in df.columns]
        if missing_curves:
            st.error(
                f"❌ This LAS file is missing required log curves: **{missing_curves}**.\n\n"
                f"The model needs: `{RAW_CURVES}`.\n\n"
                "Please upload a LAS file that contains these curves, or use a well "
                "from the Kansas Geological Survey dataset."
            )
            st.stop()

        # ── PE imputation if partially missing ────────────────────────────────
        if df['PE'].isna().any():
            df = impute_pe(df)

        # ── Feature engineering ───────────────────────────────────────────────
        df_feat = build_engineered_features(df)

        # Select ONLY the exact 16 columns the scaler/model expect — in order.
        # Any extra curves in the LAS file (CALI, SP, etc.) are safely ignored.
        X = df_feat[ENGINEERED_FEATURES]

        # ── Scale + predict ───────────────────────────────────────────────────
        model, scaler = load_model_and_scaler()
        X_scaled = scaler.transform(X)

        df_feat["Pred_Facies"]    = model.predict(X_scaled)
        df_feat["Pred_Lithology"] = df_feat["Pred_Facies"].map(FACIES_LABELS)

        # ── Metrics sidebar ───────────────────────────────────────────────────
        facies_counts = df_feat["Pred_Facies"].value_counts().sort_index()
        dominant = df_feat["Pred_Facies"].value_counts().idxmax()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total depth samples", f"{len(df_feat):,}")
        col2.metric("Dominant facies", f"{dominant} — {FACIES_LABELS[dominant]}")
        col3.metric("Unique facies predicted", facies_counts.shape[0])

        # ── Facies distribution bar chart ─────────────────────────────────────
        st.subheader("Predicted Facies Distribution")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 3))
        bar_labels = [f"{i}\n{FACIES_LABELS[i][:12]}" for i in facies_counts.index]
        bar_colors  = [FACIES_COLORS[i - 1] for i in facies_counts.index]
        ax_bar.bar(bar_labels, facies_counts.values, color=bar_colors, edgecolor='white')
        ax_bar.set_ylabel("Sample count")
        ax_bar.set_title("Number of depth samples per predicted facies class")
        ax_bar.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_bar)

        # ── Well log track (4 tracks) ─────────────────────────────────────────
        st.subheader("Well Log Track")
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 12), sharey=True)

        depth = df_feat["Depth"]

        # Track 1: Gamma Ray
        axes[0].plot(df_feat["GR"], depth, color='black', lw=1.0)
        axes[0].set_title("Gamma Ray\n(GR)", fontsize=10)
        axes[0].set_xlabel("API units")
        axes[0].grid(True, linestyle=':', alpha=0.5)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Depth (m / ft)")

        # Track 2: Resistivity
        axes[1].plot(df_feat["ILD_log10"], depth, color='steelblue', lw=1.0)
        axes[1].set_title("Resistivity\n(ILD log10)", fontsize=10)
        axes[1].set_xlabel("Log10 Ohm·m")
        axes[1].grid(True, linestyle=':', alpha=0.5)

        # Track 3: Porosity (PHIND)
        axes[2].plot(df_feat["PHIND"], depth, color='seagreen', lw=1.0)
        axes[2].set_title("Porosity\n(PHIND)", fontsize=10)
        axes[2].set_xlabel("Decimal fraction")
        axes[2].grid(True, linestyle=':', alpha=0.5)

        # Track 4: Predicted facies colour strip
        pred_strip = np.repeat(df_feat["Pred_Facies"].values, 50).reshape(-1, 50)
        axes[3].imshow(
            pred_strip, cmap=CMAP_FACIES, aspect="auto",
            extent=[0, 10, depth.max(), depth.min()],
            vmin=1, vmax=9,
        )
        axes[3].set_title("Predicted\nFacies", fontsize=10)
        axes[3].set_xticks([])

        # Facies colour legend below the track plot
        legend_patches = facies_legend_patches()
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            ncol=3,
            fontsize=8,
            title="Facies Classes",
            title_fontsize=9,
            bbox_to_anchor=(0.5, -0.06),
            frameon=True,
        )

        plt.suptitle(f"Well Log Interpretation — {uploaded_file.name}", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)

        # ── Data table ────────────────────────────────────────────────────────
        with st.expander("View prediction table (first 50 rows)"):
            st.dataframe(
                df_feat[["Depth", "GR", "ILD_log10", "PHIND", "Pred_Facies", "Pred_Lithology"]]
                .head(50)
                .reset_index(drop=True)
            )

        # ── Download ──────────────────────────────────────────────────────────
        csv_out = df_feat[["Depth", "Pred_Facies", "Pred_Lithology"]].to_csv(index=False)
        st.download_button(
            label="⬇️ Download predictions as CSV",
            data=csv_out,
            file_name="facies_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Could not process this file: {e}")

else:
    st.info(
        "👆 Upload a `.las` file with **GR, ILD_log10, DeltaPHI, PHIND, and PE** "
        "curves to get started. Extra curves in the file are safely ignored."
    )

st.markdown("---")
st.caption(
    "Model: Random Forest (tuned via GridSearchCV, trained on LAS-available curves only) · "
    "Dataset: SEG 2016 ML Contest / Kansas Geological Survey, Council Grove Field · "
    "Built as a BYOP internship portfolio project."
)
