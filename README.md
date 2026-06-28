# Well-Log Facies Prediction (ML-Based Lithology Classification)

A multi-class machine learning pipeline that predicts rock facies from wireline well-log
measurements — built as a BYOP (Build Your Own Project) internship portfolio piece bridging
**geophysics and machine learning**.

🔗 **Live demo:** [Streamlit app link here]
📓 **Notebook:** [`BYOP_updated.ipynb`](./BYOP_updated.ipynb)

---

## Problem Statement

Given wireline log measurements (gamma ray, resistivity, porosity logs) recorded at half-foot
depth intervals, predict the rock facies (lithology class) at each depth — a task normally
done by manually inspecting rock cores. Automating this with ML can reduce the cost and time
of subsurface characterization in oil & gas and groundwater applications.

## Dataset

- **Source:** [SEG 2016 Machine Learning Contest](https://github.com/seg/2016-ml-contest), originally
  from the Kansas Geological Survey — Council Grove Field, Hugoton & Panoma gas reservoirs,
  Southwest Kansas.
- **9 wells, 3232 labeled samples, 9 facies classes.**
- **Features:** GR (gamma ray), ILD_log10 (resistivity), DeltaPHI, PHIND (porosity logs), PE
  (photoelectric effect), plus two geologic indicator variables (NM_M, RELPOS).

## Methodology

1. **EDA** — missingness check, distribution analysis, petrophysical crossplots, class
   imbalance quantification.
2. **Preprocessing** — per-well median imputation (with global fallback), StandardScaler
   normalization (fit on train only to avoid leakage), stratified 70/15/15 train/val/test split.
3. **Feature engineering** — GR/PHIND ratio (lithology-porosity proxy), rolling mean/std over a
   3-sample window computed **per well** to avoid cross-well bleed.
4. **Modeling** — Random Forest and XGBoost (GridSearchCV-tuned), plus a 1D-CNN trained on
   depth-windowed sequences (window size 5) with class-weighted loss to handle imbalance.
5. **Explainability** — SHAP summary plots on the Random Forest to identify which logs drive
   predictions.
6. **Evaluation** — final test-set classification report, confusion matrix, and a head-to-head
   comparison table across all three models.
7. **Deployment** — a Streamlit app that accepts a raw `.las` file and outputs a predicted
   facies log track.

## Results

| Model | Test Accuracy | Test Macro F1 |
|---|---|---|
| Random Forest | ~0.70 | ~0.69 |
| XGBoost | ~0.70 | ~0.69 |
| 1D-CNN | see `model_comparison.csv` | see `model_comparison.csv` |

Facies 5 was the hardest class to classify (test recall dropped sharply versus validation),
suggesting genuine petrophysical overlap with a neighboring lithology rather than a simple
data-scarcity issue. Full discussion in the notebook's **Final Results Summary** section.

## Repository Structure

```
.
├── BYOP_updated.ipynb        # Full pipeline: EDA → preprocessing → modeling → evaluation
├── app.py                    # Streamlit app for live facies prediction from a LAS file
├── best_baseline_rf.joblib   # Trained Random Forest model checkpoint
├── scaler.joblib             # Fitted StandardScaler
├── model_comparison.csv      # RF vs XGBoost vs CNN test metrics
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
shap
lasio
streamlit
joblib
```

## Domain Tags

`geophysics` `petrophysics` `machine-learning` `random-forest` `xgboost` `deep-learning`
`explainable-ai` `well-logs` `lithology-classification`

## Acknowledgements

Dataset originally compiled by Brendon Hall for the SEG 2016 Machine Learning Contest, sourced
from the Kansas Geological Survey. Released under CC-BY for use on this problem.
