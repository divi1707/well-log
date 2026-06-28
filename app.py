
"""
Well-Log Facies Prediction — Streamlit App
--------------------------------------------
Upload a LAS file (or use the demo well), predict facies down the depth
column using the trained Random Forest model, and view the result as a
well log track.

Run locally:
    streamlit run app.py

Requires these files in the same folder (produced by the training notebook):
    best_baseline_rf.joblib   - trained Random Forest model
    scaler.joblib             - fitted StandardScaler
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

FACIES_LABELS = {
    1: "Nonmarine sandstone", 2: "Nonmarine coarse siltstone",
    3: "Nonmarine fine siltstone", 4: "Marine siltstone & shale",
    5: "Mudstone", 6: "Wackestone", 7: "Dolomite",
    8: "Packstone-grainstone", 9: "Bafflestone",
}
FACIES_COLORS = ['#F4D03F', '#F5B041', '#DC7633', '#A11D33', '#1B4F72',
                  '#2E4053', '#7D6608', '#117A65', '#145A32']
CMAP_FACIES = mcolors.ListedColormap(FACIES_COLORS, 'indexed')

REQUIRED_FEATURES = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("best_baseline_rf.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


def build_engineered_features(df):
    """Recreate the same feature engineering used in training."""
    df = df.copy()
    df['GR_PHIND_ratio'] = df['GR'] / (df['PHIND'] + 0.001)
    core_curves = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    for curve in core_curves:
        df[f'{curve}_roll_mean'] = df[curve].rolling(window=3, min_periods=1).mean()
        df[f'{curve}_roll_std'] = df[curve].rolling(window=3, min_periods=1).std().fillna(0)
    return df


uploaded_file = st.file_uploader("Upload a LAS file", type=["las"])

if uploaded_file is not None:
    try:
        las_bytes = uploaded_file.read().decode("utf-8", errors="ignore")
        las = lasio.read(io.StringIO(las_bytes))
        df = las.df().reset_index()
        depth_col = df.columns[0]
        df = df.rename(columns={depth_col: "Depth"})

        st.success(f"Loaded LAS file — {len(df)} depth samples, "
                    f"{len(df.columns)-1} curves found.")
        st.write("Curves detected:", list(df.columns))

        missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
        if missing:
            st.error(
                f"This LAS file is missing required curves for the trained model: "
                f"{missing}. The model expects {REQUIRED_FEATURES}. "
                "Try the demo well instead, or upload a LAS file from the Kansas "
                "Geological Survey facies dataset."
            )
        else:
            model, scaler = load_model_and_scaler()
            df_feat = build_engineered_features(df)

            feature_cols = [c for c in df_feat.columns if c not in ("Depth",)]
            X = df_feat[feature_cols]
            X_scaled = scaler.transform(X)

            df_feat["Pred_Facies"] = model.predict(X_scaled)
            df_feat["Pred_Lithology"] = df_feat["Pred_Facies"].map(FACIES_LABELS)

            st.subheader("Predicted Facies (first 20 rows)")
            st.dataframe(df_feat[["Depth", "GR", "Pred_Facies", "Pred_Lithology"]].head(20))

            # --- Well log track plot ---
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 10), sharey=True)

            ax[0].plot(df_feat["GR"], df_feat["Depth"], color="black", lw=1.0)
            ax[0].set_title("Gamma Ray (GR)")
            ax[0].set_xlabel("API units")
            ax[0].invert_yaxis()
            ax[0].grid(True, linestyle=":", alpha=0.5)

            pred_strip = np.repeat(df_feat["Pred_Facies"].values, 50).reshape(-1, 50)
            ax[1].imshow(
                pred_strip, cmap=CMAP_FACIES, aspect="auto",
                extent=[0, 10, df_feat["Depth"].max(), df_feat["Depth"].min()],
                vmin=1, vmax=9,
            )
            ax[1].set_title("Predicted Facies")
            ax[1].set_xticks([])

            st.subheader("Well Log Track")
            st.pyplot(fig)

            csv = df_feat[["Depth", "Pred_Facies", "Pred_Lithology"]].to_csv(index=False)
            st.download_button("Download predictions as CSV", csv, "facies_predictions.csv")

    except Exception as e:
        st.error(f"Could not process this file: {e}")

else:
    st.info(
        "No file uploaded yet. Upload a `.las` file with GR, ILD_log10, DeltaPHI, "
        "PHIND, and PE curves to get started."
    )

st.markdown("---")
st.caption(
    "Model: Random Forest (tuned via GridSearchCV) · "
    "Dataset: SEG 2016 ML Contest / Kansas Geological Survey, Council Grove Field · "
    "Built as a BYOP internship portfolio project."
)
