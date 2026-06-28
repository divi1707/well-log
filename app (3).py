"""
Well-Log Facies Prediction — Streamlit App
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
    "Upload any `.las` file — the app will auto-detect your curve names, "
    "plot the raw logs, and predict facies using a Random Forest trained on "
    "the SEG 2016 / Kansas Geological Survey dataset."
)

# ── Facies metadata ────────────────────────────────────────────────────────────
FACIES_LABELS = {
    1: "Nonmarine sandstone", 2: "Nonmarine coarse siltstone",
    3: "Nonmarine fine siltstone", 4: "Marine siltstone & shale",
    5: "Mudstone", 6: "Wackestone", 7: "Dolomite",
    8: "Packstone-grainstone", 9: "Bafflestone",
}
FACIES_COLORS = ['#F4D03F','#F5B041','#DC7633','#A11D33','#1B4F72',
                 '#2E4053','#7D6608','#117A65','#145A32']
CMAP_FACIES = mcolors.ListedColormap(FACIES_COLORS, 'indexed')

RAW_CURVES = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']

ENGINEERED_FEATURES = (
    RAW_CURVES + ['GR_PHIND_ratio']
    + [f'{c}_roll_mean' for c in RAW_CURVES]
    + [f'{c}_roll_std'  for c in RAW_CURVES]
)

# ── Common aliases: maps any LAS mnemonic → our internal name ─────────────────
# Add more rows here if you encounter other naming conventions.
CURVE_ALIASES = {
    # Gamma Ray
    'GR': 'GR', 'GAM': 'GR', 'GAMMA': 'GR', 'GRD': 'GR', 'SGR': 'GR',

    # Resistivity → we need log10 of it; handled separately below
    'ILD': 'ILD_log10', 'ILD_LOG10': 'ILD_log10', 'RT': 'ILD_log10',
    'RD': 'ILD_log10', 'RESD': 'ILD_log10', 'LLD': 'ILD_log10',
    'RILD': 'ILD_log10', 'AT90': 'ILD_log10', 'M2RX': 'ILD_log10',

    # DeltaPHI (neutron-density porosity difference)
    'DELTAPHI': 'DeltaPHI', 'DPHI': 'DeltaPHI', 'DPHZ': 'DeltaPHI',
    'DNPHI': 'DeltaPHI',

    # PHIND (average porosity)
    'PHIND': 'PHIND', 'PHIA': 'PHIND', 'PHIT': 'PHIND', 'PHIE': 'PHIND',
    'NPHI': 'PHIND', 'TNPH': 'PHIND', 'CNCF': 'PHIND', 'NEUT': 'PHIND',

    # PE / Photoelectric
    'PE': 'PE', 'PEF': 'PE', 'PEFZ': 'PE',
}

# Resistivity curves that need log10 applied
RESISTIVITY_CURVES = {'ILD', 'RT', 'RD', 'RESD', 'LLD', 'RILD', 'AT90', 'M2RX'}


@st.cache_resource
def load_model_and_scaler():
    model  = joblib.load("best_baseline_rf.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


def remap_columns(df: pd.DataFrame):
    """
    Try to map LAS column names to our internal names using CURVE_ALIASES.
    Returns (remapped_df, mapping_dict, still_missing_list).
    """
    df = df.copy()
    upper_cols = {c.upper(): c for c in df.columns}  # upper → original
    mapping = {}   # original col → internal name
    already_have = set(df.columns)

    for upper, original in upper_cols.items():
        internal = CURVE_ALIASES.get(upper)
        if internal and internal not in already_have:
            # Apply log10 for raw resistivity values (typically > 1 ohm·m)
            if upper in RESISTIVITY_CURVES:
                df[internal] = np.log10(df[original].clip(lower=0.001))
            else:
                df[internal] = df[original]
            already_have.add(internal)
            mapping[original] = internal

    missing = [c for c in RAW_CURVES if c not in df.columns]
    return df, mapping, missing


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['GR_PHIND_ratio'] = df['GR'] / (df['PHIND'] + 0.001)
    for curve in RAW_CURVES:
        df[f'{curve}_roll_mean'] = df[curve].rolling(window=3, min_periods=1).mean()
        df[f'{curve}_roll_std']  = df[curve].rolling(window=3, min_periods=1).std().fillna(0)
    return df


def impute_pe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    median = df['PE'].median()
    if pd.isna(median):
        df['PE'] = 3.0
        st.warning("PE curve entirely missing — filled with 3.0. Carbonate facies predictions may be less reliable.")
    else:
        df['PE'] = df['PE'].fillna(median)
    return df


def facies_legend_patches():
    return [mpatches.Patch(color=FACIES_COLORS[i-1], label=f"{i} — {FACIES_LABELS[i]}")
            for i in range(1, 10)]


def plot_raw_logs(df: pd.DataFrame, available_curves: list, filename: str):
    """Plot whatever curves are present in the LAS file."""
    plot_curves = [c for c in available_curves if c != 'Depth'][:8]  # max 8 tracks
    n = len(plot_curves)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 10), sharey=True)
    if n == 1:
        axes = [axes]
    depth = df['Depth']
    for ax, curve in zip(axes, plot_curves):
        ax.plot(df[curve], depth, lw=0.8, color='black')
        ax.set_title(curve, fontsize=9)
        ax.set_xlabel(curve, fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(labelsize=7)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth", fontsize=9)
    plt.suptitle(f"Raw Log Curves — {filename}", fontsize=11, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader("Upload a LAS file", type=["las"])

if uploaded_file is not None:
    try:
        las_bytes = uploaded_file.read().decode("utf-8", errors="ignore")
        las = lasio.read(io.StringIO(las_bytes))
        df = las.df().reset_index()
        depth_col = df.columns[0]
        df = df.rename(columns={depth_col: "Depth"})

        st.success(
            f"✅ **{uploaded_file.name}** — "
            f"**{len(df):,}** depth samples · **{len(df.columns)-1}** curves"
        )

        raw_curve_names = [c for c in df.columns if c != 'Depth']

        with st.expander("📋 Curves detected in this LAS file"):
            st.write(raw_curve_names)

        # ── Always show raw log plot first ────────────────────────────────────
        st.subheader("Raw Log Curves")
        plot_raw_logs(df, list(df.columns), uploaded_file.name)

        # ── Remap column names ────────────────────────────────────────────────
        df, mapping, missing = remap_columns(df)

        if mapping:
            st.info(
                "🔄 **Auto-remapped curves:** "
                + ", ".join(f"`{k}` → `{v}`" for k, v in mapping.items())
            )

        # ── Handle still-missing curves ───────────────────────────────────────
        if missing:
            st.error(
                f"❌ Could not find matches for: **{missing}**\n\n"
                "Your LAS file curves are shown above. "
                "Use the manual mapper below to assign them."
            )

            st.subheader("🔧 Manual Curve Mapping")
            st.markdown("Select which column in your LAS file corresponds to each missing curve:")
            available_options = ['(skip)'] + raw_curve_names
            user_map = {}
            for needed in missing:
                chosen = st.selectbox(f"Which column is **{needed}**?", available_options, key=needed)
                if chosen != '(skip)':
                    user_map[needed] = chosen

            if st.button("Apply mapping & predict"):
                for internal, original in user_map.items():
                    if internal == 'ILD_log10':
                        df[internal] = np.log10(df[original].clip(lower=0.001))
                    else:
                        df[internal] = df[original]
                still_missing = [c for c in RAW_CURVES if c not in df.columns]
                if still_missing:
                    st.error(f"Still missing: {still_missing}. Cannot predict without these curves.")
                    st.stop()
                st.rerun()
            st.stop()

        # ── PE imputation ─────────────────────────────────────────────────────
        if df['PE'].isna().any():
            df = impute_pe(df)

        # ── Feature engineering + prediction ──────────────────────────────────
        df_feat = build_engineered_features(df)
        X = df_feat[ENGINEERED_FEATURES]

        model, scaler = load_model_and_scaler()
        X_scaled = scaler.transform(X)

        df_feat["Pred_Facies"]    = model.predict(X_scaled)
        df_feat["Pred_Lithology"] = df_feat["Pred_Facies"].map(FACIES_LABELS)

        # ── Summary metrics ───────────────────────────────────────────────────
        facies_counts = df_feat["Pred_Facies"].value_counts().sort_index()
        dominant = df_feat["Pred_Facies"].value_counts().idxmax()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total depth samples", f"{len(df_feat):,}")
        col2.metric("Dominant facies", f"{dominant} — {FACIES_LABELS[dominant]}")
        col3.metric("Unique facies predicted", facies_counts.shape[0])

        # ── Facies distribution bar chart ─────────────────────────────────────
        st.subheader("Predicted Facies Distribution")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 3))
        bar_labels = [f"{i}\n{FACIES_LABELS[i][:14]}" for i in facies_counts.index]
        bar_colors  = [FACIES_COLORS[i-1] for i in facies_counts.index]
        ax_bar.bar(bar_labels, facies_counts.values, color=bar_colors, edgecolor='white')
        ax_bar.set_ylabel("Sample count")
        ax_bar.set_title("Depth samples per predicted facies class")
        ax_bar.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_bar)

        # ── Well log track (4 tracks + facies strip) ──────────────────────────
        st.subheader("Well Log Interpretation Track")
        fig, axes = plt.subplots(1, 5, figsize=(16, 12), sharey=True)
        depth = df_feat["Depth"]

        axes[0].plot(df_feat["GR"],       depth, color='black',    lw=1.0)
        axes[0].set_title("Gamma Ray\n(GR)", fontsize=10)
        axes[0].set_xlabel("API units"); axes[0].set_ylabel("Depth")
        axes[0].invert_yaxis(); axes[0].grid(True, linestyle=':', alpha=0.5)

        axes[1].plot(df_feat["ILD_log10"], depth, color='steelblue', lw=1.0)
        axes[1].set_title("Resistivity\n(ILD log10)", fontsize=10)
        axes[1].set_xlabel("Log10 Ohm·m"); axes[1].grid(True, linestyle=':', alpha=0.5)

        axes[2].plot(df_feat["PHIND"],    depth, color='seagreen',  lw=1.0)
        axes[2].set_title("Porosity\n(PHIND)", fontsize=10)
        axes[2].set_xlabel("p.u."); axes[2].grid(True, linestyle=':', alpha=0.5)

        axes[3].plot(df_feat["PE"],       depth, color='tomato',    lw=1.0)
        axes[3].set_title("Photoelectric\n(PE)", fontsize=10)
        axes[3].set_xlabel("b/e"); axes[3].grid(True, linestyle=':', alpha=0.5)

        pred_strip = np.repeat(df_feat["Pred_Facies"].values, 50).reshape(-1, 50)
        axes[4].imshow(pred_strip, cmap=CMAP_FACIES, aspect="auto",
                       extent=[0, 10, depth.max(), depth.min()], vmin=1, vmax=9)
        axes[4].set_title("Predicted\nFacies", fontsize=10)
        axes[4].set_xticks([])

        fig.legend(handles=facies_legend_patches(), loc='lower center', ncol=3,
                   fontsize=8, title="Facies Classes", title_fontsize=9,
                   bbox_to_anchor=(0.5, -0.07), frameon=True)
        plt.suptitle(f"Well Log Interpretation — {uploaded_file.name}", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)

        # ── Data table + download ─────────────────────────────────────────────
        with st.expander("View prediction table (first 50 rows)"):
            st.dataframe(
                df_feat[["Depth","GR","ILD_log10","PHIND","PE","Pred_Facies","Pred_Lithology"]]
                .head(50).reset_index(drop=True)
            )

        csv_out = df_feat[["Depth","Pred_Facies","Pred_Lithology"]].to_csv(index=False)
        st.download_button("⬇️ Download predictions as CSV", csv_out,
                           "facies_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Could not process this file: {e}")

else:
    st.info("👆 Upload any `.las` file — the app will auto-detect curve names and show all raw logs.")

st.markdown("---")
st.caption(
    "Model: Random Forest · Dataset: SEG 2016 ML Contest / Kansas Geological Survey · "
    "Built as a BYOP internship portfolio project."
)
