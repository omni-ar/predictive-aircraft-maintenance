# Predictive Aircraft Engine Maintenance

### Remaining Useful Life (RUL) Estimation — NASA CMAPSS (FD001)

---

## Overview

Aircraft engines degrade gradually over operational cycles. Accurate estimation of this degradation enables condition-based maintenance, reduces unexpected failures, and improves operational safety.

This project builds a structured and reproducible machine learning pipeline to estimate the **Remaining Useful Life (RUL)** of turbofan engines using multivariate sensor telemetry from NASA's CMAPSS simulator.

The focus is not only on model accuracy, but on:

- Degradation-aware feature engineering
- Engine-level validation to prevent data leakage
- Regime-based error diagnostics
- Reliability-focused evaluation using the NASA asymmetric scoring metric
- Model robustness and interpretability

---

## Problem Statement

Each engine runs from a healthy state until failure. At any given cycle:

```
RUL = Max cycles of that engine − Current cycle
```

This is a **supervised regression problem** on multivariate time-series data. The model must learn how degradation manifests in sensor patterns and translate that into a cycle-level failure estimate.

---

## Dataset

**Source:** [NASA CMAPSS — Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

Currently using subset **FD001**:

| Property             | Detail                         |
| -------------------- | ------------------------------ |
| Training engines     | 100 (run-to-failure)           |
| Test engines         | 100 (truncated before failure) |
| Operating conditions | Single                         |
| Fault mode           | HPC degradation                |
| Sensor channels      | 21                             |
| Operational settings | 3                              |

> Raw data is not tracked in Git. Place files under `data/raw/`.

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

---

## Technical Focus Areas

- Predictive Maintenance
- Time-Series Feature Engineering
- Regression-based Life Estimation
- Degradation Signal Extraction
- Model Interpretability and Reliability Diagnostics

---

## Project Architecture

### Phase 1 — Exploratory Data Analysis

- Loaded raw dataset and assigned structured column names
- Computed per-engine RUL targets
- Removed constant and near-zero variance sensors
- Analyzed degradation-sensitive sensors via RUL correlation
- Visualized cross-engine degradation trajectories
- Verified dataset integrity and absence of leakage

**Deliverable:** `notebooks/01_eda.ipynb`

---

### Phase 2 — Degradation-Aware Feature Engineering

Raw sensor values are insufficient for predictive maintenance. The model needs to observe _how_ sensors change, not just their instantaneous readings.

Implemented:

- Rolling mean — noise reduction and trend smoothing
- Rolling standard deviation — instability and variance detection
- Lag features — temporal context
- Baseline-adjusted sensors — per-engine normalization relative to initial state
- RUL capping at 130 cycles — stabilizes early-life prediction bias

Final engineered dataset: **~48 features per cycle** with engine-aware transformations.

**Deliverable:** `notebooks/02_feature_engineering.ipynb`

---

### Phase 3 — Baseline & Tuned Modeling

Models trained:

- Random Forest Regressor
- XGBoost Regressor

**Validation strategy:** 5-Fold Engine-Level `GroupKFold`  
Engines are split at the unit level — no engine appears in both train and validation. This prevents cross-engine data leakage and ensures the model generalizes to unseen engines.

**Results:**

| Model                                     | Validation RMSE     |
| ----------------------------------------- | ------------------- |
| Random Forest (v3 features, single split) | ~13.07 cycles       |
| XGBoost (v3 features, single split)       | ~13.55 cycles       |
| Random Forest (5-Fold GroupKFold)         | 15.18 ± 1.25 cycles |

The GroupKFold result is the honest estimate — single-split RMSE is optimistic.

**Deliverable:** `notebooks/03_model_baseline.ipynb`

---

### Phase 4 — Advanced Evaluation & Reliability Diagnostics

Beyond RMSE, the project evaluates model behavior from a reliability engineering perspective.

**NASA Scoring Metric** — asymmetric penalty function that penalizes late predictions (overestimating RUL near failure) more severely than early predictions.

**Regime-Based Error Analysis** — engines segmented by lifecycle stage:

| Regime       | Cycles Remaining | Finding               |
| ------------ | ---------------- | --------------------- |
| Near Failure | 0–30             | Strong performance    |
| Mid Life     | 30–80            | Mild optimism bias    |
| Early Life   | 80–130           | Slight pessimism bias |

**Feature Ablation** — removed `sensor_4_roll_mean` (dominant feature at 0.558 importance). RMSE increased modestly, confirming model robustness and that remaining features carry complementary signal.

---

### Phase 5 — Sequence Modeling Experiment

Implemented an LSTM-based regression pipeline with:

- 30-cycle sliding window framing
- Proper input and target scaling
- Engine-level fold validation

**Result:** Tree-based models outperformed vanilla LSTM under FD001 conditions.

**Conclusion:** Engineered tabular features are more effective at this dataset scale and fault regime. Sequence modeling may add value on multi-condition subsets (FD002–FD004).

**Deliverable:** `notebooks/04_sequence_modeling.ipynb`

---

### Phase 6 — Cost-Sensitive Learning (Exploratory)

Applied sample weighting to emphasize near-failure regime during training.

**Outcome:** Marginal NASA score improvement, no significant RMSE gain. Indicates that degradation-aware features already capture late-stage behavior effectively.

---

## Final Model Summary

| Component   | Description                                                 |
| ----------- | ----------------------------------------------------------- |
| Model       | Tuned Random Forest                                         |
| Features    | 48 engineered degradation-aware features                    |
| Validation  | 5-Fold Engine-Level GroupKFold                              |
| Metrics     | RMSE, MAE, NASA asymmetric score                            |
| Diagnostics | Regime error analysis, feature ablation, importance ranking |

---

## Key Technical Insights

- Engine-level splitting is mandatory — row-level splits cause severe data leakage
- RUL capping at 130 cycles stabilizes early-life prediction variance
- Rolling degradation features consistently outperform raw sensor values
- Tree ensembles outperform vanilla LSTM on FD001 at this scale
- Regime-based error analysis reveals systematic bias invisible in aggregate RMSE
- Feature redundancy across engineered signals increases model robustness

---

## Repository Structure

```
predictive-aircraft-maintenance/
│
├── data/
│   ├── raw/                        # Original NASA files (not tracked in Git)
│   └── processed/                  # Engineered feature exports (not tracked in Git)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   └── 04_sequence_modeling.ipynb
│
├── models/                         # Saved model artifacts
├── outputs/                        # Plots and evaluation outputs
├── src/
│   └── predict.py                  # Inference pipeline (optional extension)
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/omni-ar/predictive-aircraft-maintenance.git
cd predictive-aircraft-maintenance
pip install -r requirements.txt
```

Download the CMAPSS dataset from NASA and place under `data/raw/`.

---

## Future Extensions

- LightGBM comparison
- Hyperparameter optimization via Optuna
- Prediction uncertainty estimation via conformal prediction intervals
- Deployment-ready inference API

---

## Evaluation Metrics

| Metric         | Description                                                      |
| -------------- | ---------------------------------------------------------------- |
| **RMSE**       | Penalizes large errors — sensitive to late-life mispredictions   |
| **MAE**        | Direct average cycle-level error                                 |
| **NASA Score** | Asymmetric — overestimation near failure penalized exponentially |

---

## Development Principles

- Engine-level validation is non-negotiable
- Understand degradation before engineering features
- Baseline before complexity — earn every upgrade
- Interpretability and diagnostics matter as much as accuracy

---

## Author

**Arjit Tripathi**  
B.Tech Computer Science & Engineering — Vellore Institute of Technology
