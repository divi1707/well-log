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
st.title("Well-Log Facies Prediction")
st.markdown(
    "Upload any `.las` file — the app will auto-detect your curve names, "
    "plot the raw logs, and predict facies using a Random Forest trained on "
    "the SEG 2016 / Kansas Geological Survey dataset."
)

# ── Facies metadata ────────────────────────────────────────────────────────────
# Each facies has a distinct, geologically-meaningful colour:
#   1  Nonmarine sandstone        — warm yellow      (clean, high-GR sand)
#   2  Nonmarine coarse siltstone — amber            (coarser clastic)
#   3  Nonmarine fine siltstone   — burnt sienna     (finer clastic)
#   4  Marine siltstone & shale   — deep burgundy    (marine, clay-rich)
#   5  Mudstone                   — slate blue       (low energy, argillaceous)
#   6  Wackestone                 — medium teal      (mud-supported carbonate)
#   7  Dolomite                   — bright cyan      (diagenetic, distinctive)
#   8  Packstone-grainstone       — olive green      (grain-supported carbonate)
#   9  Bafflestone                — dark forest      (reef/organic buildup)
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
    "#F9E04B",   # 1 — Nonmarine sandstone        warm yellow
    "#E8923A",   # 2 — Nonmarine coarse siltstone amber
    "#C1572B",   # 3 — Nonmarine fine siltstone   burnt sienna
    "#8B1A2F",   # 4 — Marine siltstone & shale   deep burgundy
    "#4A6FA5",   # 5 — Mudstone                   slate blue
    "#2E9E8E",   # 6 — Wackestone                 medium teal
    "#45C4D4",   # 7 — Dolomite                   bright cyan
    "#7A9E3B",   # 8 — Packstone-grainstone       olive green
    "#2D5A27",   # 9 — Bafflestone                dark forest green
]
CMAP_FACIES = mcolors.ListedColormap(FACIES_COLORS, "indexed")

# Distinct colours for the raw log curve tracks
CURVE_TRACK_COLORS = {
    "GR":       "#2C3E50",   # near-black        (standard industry GR colour)
    "ILD_log10":"#1A6BAD",   # strong blue       (resistivity)
    "DeltaPHI": "#8E44AD",   # purple            (density-neutron difference)
    "PHIND":    "#16A085",   # teal-green        (porosity)
    "PE":       "#C0392B",   # deep red          (photoelectric)
}
# Fallback colour palette for extra LAS curves beyond the 5 expected ones
EXTRA_CURVE_COLORS = [
    "#D4AC0D", "#117A65", "#6C3483", "#1A5276",
    "#784212", "#1B2631", "#922B21", "#0E6655",
]

RAW_CURVES = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]

ENGINEERED_FEATURES = (
    RAW_CURVES + ["GR_PHIND_ratio"]
    + [f"{c}_roll_mean" for c in RAW_CURVES]
    + [f"{c}_roll_std"  for c in RAW_CURVES]
)

# ── Common aliases ─────────────────────────────────────────────────────────────
CURVE_ALIASES = {
    "GR": "GR", "GAM": "GR", "GAMMA": "GR", "GRD": "GR", "SGR": "GR",
    "ILD": "ILD_log10", "ILD_LOG10": "ILD_log10", "RT": "ILD_log10",
    "RD": "ILD_log10", "RESD": "ILD_log10", "LLD": "ILD_log10",
    "RILD": "ILD_log10", "AT90": "ILD_log10", "M2RX": "ILD_log10",
    "DELTAPHI": "DeltaPHI", "DPHI": "DeltaPHI", "DPHZ": "DeltaPHI", "DNPHI": "DeltaPHI",
    "PHIND": "PHIND", "PHIA": "PHIND", "PHIT": "PHIND", "PHIE": "PHIND",
    "NPHI": "PHIND", "TNPH": "PHIND", "CNCF": "PHIND", "NEUT": "PHIND",
    "PE": "PE", "PEF": "PE", "PEFZ": "PE",
}
RESISTIVITY_CURVES = {"ILD", "RT", "RD", "RESD", "LLD", "RILD", "AT90", "M2RX"}


@st.cache_resource
def load_model_and_scaler():
    model  = joblib.load("best_baseline_rf.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


def remap_columns(df):
    df = df.copy()
    upper_cols = {c.upper(): c for c in df.columns}
    mapping = {}
    already_have = set(df.columns)
    for upper, original in upper_cols.items():
        internal = CURVE_ALIASES.get(upper)
        if internal and internal not in already_have:
            if upper in RESISTIVITY_CURVES:
                df[internal] = np.log10(df[original].clip(lower=0.001))
            else:
                df[internal] = df[original]
            already_have.add(internal)
            mapping[original] = internal
    missing = [c for c in RAW_CURVES if c not in df.columns]
    return df, mapping, missing


def build_engineered_features(df):
    df = df.copy()
    df["GR_PHIND_ratio"] = df["GR"] / (df["PHIND"] + 0.001)
    for curve in RAW_CURVES:
        df[f"{curve}_roll_mean"] = df[curve].rolling(window=3, min_periods=1).mean()
        df[f"{curve}_roll_std"]  = df[curve].rolling(window=3, min_periods=1).std().fillna(0)
    return df


def impute_pe(df):
    df = df.copy()
    median = df["PE"].median()
    if pd.isna(median):
        df["PE"] = 3.0
        st.warning(
            "PE curve is entirely missing. Filled with 3.0 — "
            "carbonate facies predictions may be less reliable."
        )
    else:
        df["PE"] = df["PE"].fillna(median)
    return df


def facies_legend_patches():
    return [
        mpatches.Patch(color=FACIES_COLORS[i - 1], label=f"{i}  {FACIES_LABELS[i]}")
        for i in range(1, 10)
    ]


def plot_raw_logs(df, filename):
    """Plot all curves from the LAS file, each in its own distinct colour."""
    plot_curves = [c for c in df.columns if c != "Depth"][:10]
    n = len(plot_curves)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 11), sharey=True)
    if n == 1:
        axes = [axes]
    depth = df["Depth"]
    extra_idx = 0
    for ax, curve in zip(axes, plot_curves):
        # Pick colour: known curve → fixed colour, unknown → cycle through extras
        if curve in CURVE_TRACK_COLORS:
            color = CURVE_TRACK_COLORS[curve]
        else:
            color = EXTRA_CURVE_COLORS[extra_idx % len(EXTRA_CURVE_COLORS)]
            extra_idx += 1
        ax.plot(df[curve], depth, lw=0.9, color=color)
        ax.set_title(curve, fontsize=9, color=color, fontweight="bold")
        ax.set_xlabel(curve, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=7)
        # Shade fill from curve to left axis for visual clarity
        ax.fill_betweenx(depth, df[curve], df[curve].min(),
                         alpha=0.08, color=color)
        # Spine colour to match curve
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth", fontsize=9)
    fig.patch.set_facecolor("#F8F9FA")
    plt.suptitle(f"Raw Log Curves — {filename}", fontsize=12, y=1.02, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)


def plot_interpretation_track(df_feat, filename):
    """5-track plot: GR, Resistivity, Porosity, PE, Predicted Facies strip."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 13), sharey=True)
    depth = df_feat["Depth"]

    track_specs = [
        ("GR",        "Gamma Ray (GR)",          "API units",    CURVE_TRACK_COLORS["GR"]),
        ("ILD_log10", "Resistivity (ILD log10)", "Log10 Ohm·m",  CURVE_TRACK_COLORS["ILD_log10"]),
        ("PHIND",     "Porosity (PHIND)",         "p.u.",         CURVE_TRACK_COLORS["PHIND"]),
        ("PE",        "Photoelectric (PE)",       "b/e",          CURVE_TRACK_COLORS["PE"]),
    ]

    for ax, (col, title, xlabel, color) in zip(axes[:4], track_specs):
        ax.plot(df_feat[col], depth, color=color, lw=1.1)
        ax.fill_betweenx(depth, df_feat[col], df_feat[col].min(),
                         alpha=0.10, color=color)
        ax.set_title(title, fontsize=9, color=color, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)

    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth", fontsize=9)

    # Facies colour strip
    pred_strip = np.repeat(df_feat["Pred_Facies"].values, 50).reshape(-1, 50)
    axes[4].imshow(
        pred_strip, cmap=CMAP_FACIES, aspect="auto",
        extent=[0, 10, depth.max(), depth.min()],
        vmin=1, vmax=9,
    )
    axes[4].set_title("Predicted Facies", fontsize=9, fontweight="bold")
    axes[4].set_xticks([])
    for spine in axes[4].spines.values():
        spine.set_edgecolor("#555")

    # Legend
    fig.legend(
        handles=facies_legend_patches(),
        loc="lower center", ncol=3,
        fontsize=8, title="Facies Classes", title_fontsize=9,
        bbox_to_anchor=(0.5, -0.08), frameon=True,
    )
    fig.patch.set_facecolor("#F8F9FA")
    plt.suptitle(f"Well Log Interpretation — {filename}", fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)


def plot_facies_bar(df_feat):
    facies_counts = df_feat["Pred_Facies"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    bar_labels = [f"{i}\n{FACIES_LABELS[i]}" for i in facies_counts.index]
    bar_colors  = [FACIES_COLORS[i - 1] for i in facies_counts.index]
    bars = ax.bar(bar_labels, facies_counts.values, color=bar_colors,
                  edgecolor="white", linewidth=0.8)
    # Value labels on bars
    for bar, val in zip(bars, facies_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_ylabel("Depth sample count", fontsize=9)
    ax.set_title("Predicted Facies Distribution", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")
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
            f"Loaded {uploaded_file.name} — "
            f"{len(df):,} depth samples, {len(df.columns) - 1} curves found."
        )

        raw_curve_names = [c for c in df.columns if c != "Depth"]
        with st.expander("Curves detected in this LAS file"):
            st.write(raw_curve_names)

        # Always show raw logs first
        st.subheader("Raw Log Curves")
        plot_raw_logs(df, uploaded_file.name)

        # Remap
        df, mapping, missing = remap_columns(df)
        if mapping:
            st.info(
                "Auto-remapped curves: "
                + ", ".join(f"`{k}` to `{v}`" for k, v in mapping.items())
            )

        # Manual mapper if still missing
        if missing:
            st.error(
                f"Could not find matches for: {missing}. "
                "Use the manual mapper below to assign them."
            )
            st.subheader("Manual Curve Mapping")
            st.markdown("Select which column in your LAS file corresponds to each missing curve:")
            available_options = ["(skip)"] + raw_curve_names
            user_map = {}
            for needed in missing:
                chosen = st.selectbox(f"Which column is {needed}?", available_options, key=needed)
                if chosen != "(skip)":
                    user_map[needed] = chosen
            if st.button("Apply mapping and predict"):
                for internal, original in user_map.items():
                    if internal == "ILD_log10":
                        df[internal] = np.log10(df[original].clip(lower=0.001))
                    else:
                        df[internal] = df[original]
                still_missing = [c for c in RAW_CURVES if c not in df.columns]
                if still_missing:
                    st.error(f"Still missing: {still_missing}. Cannot predict without these curves.")
                    st.stop()
                st.rerun()
            st.stop()

        # PE imputation
        if df["PE"].isna().any():
            df = impute_pe(df)

        # Feature engineering + prediction
        df_feat = build_engineered_features(df)
        X = df_feat[ENGINEERED_FEATURES]
        model, scaler = load_model_and_scaler()
        X_scaled = scaler.transform(X)
        df_feat["Pred_Facies"]    = model.predict(X_scaled)
        df_feat["Pred_Lithology"] = df_feat["Pred_Facies"].map(FACIES_LABELS)

        # Summary metrics
        facies_counts = df_feat["Pred_Facies"].value_counts().sort_index()
        dominant = df_feat["Pred_Facies"].value_counts().idxmax()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total depth samples", f"{len(df_feat):,}")
        col2.metric("Dominant facies", f"{dominant} — {FACIES_LABELS[dominant]}")
        col3.metric("Unique facies predicted", facies_counts.shape[0])

        # Facies colour key table
        st.subheader("Facies Colour Key")
        colour_df = pd.DataFrame([
            {"Facies": i, "Lithology": FACIES_LABELS[i], "Hex colour": FACIES_COLORS[i - 1]}
            for i in range(1, 10)
        ])

        def colour_cell(val):
            return f"background-color: {val}; color: {'#111' if val in ['#F9E04B','#E8923A','#45C4D4'] else '#fff'};"

        st.dataframe(
            colour_df.style.applymap(colour_cell, subset=["Hex colour"]),
            use_container_width=True, hide_index=True,
        )

        # Facies distribution bar chart
        st.subheader("Predicted Facies Distribution")
        plot_facies_bar(df_feat)

        # Well log interpretation track
        st.subheader("Well Log Interpretation Track")
        plot_interpretation_track(df_feat, uploaded_file.name)

        # Data table
        with st.expander("View prediction table (first 50 rows)"):
            st.dataframe(
                df_feat[["Depth", "GR", "ILD_log10", "PHIND", "PE",
                          "Pred_Facies", "Pred_Lithology"]]
                .head(50).reset_index(drop=True)
            )

        # Download
        csv_out = df_feat[["Depth", "Pred_Facies", "Pred_Lithology"]].to_csv(index=False)
        st.download_button(
            "Download predictions as CSV", csv_out,
            "facies_predictions.csv", mime="text/csv",
        )

    except Exception as e:
        st.error(f"Could not process this file: {e}")

else:
    st.info(
        "Upload any .las file — the app will auto-detect curve names and show all raw logs."
    )

st.markdown("---")
st.caption(
    "Model: Random Forest (tuned via GridSearchCV) · "
    "Dataset: SEG 2016 ML Contest / Kansas Geological Survey, Council Grove Field · "
    "Built as a BYOP internship portfolio project."
)
