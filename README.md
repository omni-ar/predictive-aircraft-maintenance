# Predictive Aircraft Engine Maintenance

### Remaining Useful Life (RUL) Estimation — NASA CMAPSS Dataset

---

## Overview

Aircraft engines experience gradual performance degradation over operational cycles. Accurate estimation of this degradation enables condition-based maintenance and reduces the risk of unexpected failure.

This project builds a machine learning pipeline to estimate the **Remaining Useful Life (RUL)** of aircraft engines from multivariate sensor telemetry. The dataset comes from NASA's CMAPSS simulator, which models engine degradation under realistic fault conditions.

The objective is to develop a structured and reproducible predictive maintenance pipeline — from data understanding to model evaluation.

---

## Problem Statement

Each engine in the dataset runs from a healthy state until failure. At any given cycle:

```
RUL = Max cycles of that engine − Current cycle
```

This is a **supervised regression problem** on multivariate time-series data. The model must learn how sensor patterns change as an engine ages, and translate that into a cycle estimate.

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

> Raw data files are not tracked in this repository. Download from the NASA link above and place under `data/raw/`.

---

## Technical Focus Areas

- Predictive Maintenance
- Time-Series Modeling
- Regression-based Life Estimation
- Feature Engineering for Degradation Signals
- Model Interpretability

---

## Project Execution Roadmap

### Phase 1 — Exploratory Data Analysis _(in progress)_

The goal here is to understand the data before touching a model. Specifically:

- Load the raw dataset and assign column names
- Identify and remove constant or near-zero-variance sensors
- Compute RUL labels for the training set
- Analyze how each sensor behaves as an engine approaches failure
- Visualize sensor drift across engine lifecycles
- Identify which sensors are actually informative vs. noise

**What to do right now (EDA checklist):**

```
[ ] Load FD001 train and test files, assign column names
[ ] Drop sensor columns with zero or near-constant variance
[ ] Plot raw sensor values for 3-5 individual engines
[ ] Overlay multiple engines on the same sensor to spot patterns
[ ] Compute correlation of each sensor with RUL
[ ] Rank sensors by degradation signal strength
[ ] Check for missing values, duplicate rows, anomalous cycles
[ ] Plot RUL distribution across training engines
[ ] Save cleaned dataframe for Phase 2
```

Deliverable: `notebooks/01_eda.ipynb`

---

### Phase 2 — Feature Engineering

Raw sensor readings alone are not enough. The model needs to see degradation dynamics, not just instantaneous values.

- Normalize sensor values (MinMax or StandardScaler)
- Create rolling window statistics: mean, std over last N cycles
- Create lag features: sensor value at t-1, t-2, t-k
- Compute degradation slope per engine per sensor
- Aggregate temporal behavior into per-engine summaries

Deliverable: `notebooks/02_feature_engineering.ipynb`

---

### Phase 3 — Baseline Modeling

Train first, tune later. Establish a clean baseline before adding complexity.

- Random Forest regressor as the initial baseline
- XGBoost as the second baseline
- Evaluate both with RMSE and MAE
- Examine error distribution — where does the model fail?

Deliverable: `notebooks/03_model_baseline.ipynb`

---

### Phase 4 — Model Improvement

- Hyperparameter tuning (GridSearch / Optuna)
- Feature importance analysis — which engineered features matter most
- Stratified error analysis: early-life vs late-life prediction quality
- Optional: LSTM or GRU for sequence-aware modeling

---

### Phase 5 — Deployment Simulation _(optional)_

- Serialize the final model
- Build a lightweight inference pipeline
- Simulate prediction on a new incoming engine stream

Deliverable: `models/`, `src/predict.py`

---

## Repository Structure

```
predictive-aircraft-maintenance/
│
├── data/
│   └── raw/                    # Download dataset here (not tracked in Git)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_baseline.ipynb
│
├── src/
│   └── predict.py              # Inference pipeline (Phase 5)
│
├── models/                     # Saved model artifacts
├── outputs/                    # Plots, results, exports
│
├── requirements.txt
└── README.md
```

---

## Evaluation Metrics

| Metric   | Why it matters                                 |
| -------- | ---------------------------------------------- |
| **RMSE** | Penalizes large prediction errors more heavily |
| **MAE**  | Gives a direct average cycle-level error       |

Both metrics are in units of **engine cycles**, which makes them directly interpretable in an operational context.

---

## Setup

```bash
git clone https://github.com/omni-ar/predictive-aircraft-maintenance.git
cd predictive-aircraft-maintenance
pip install -r requirements.txt
```

Download the CMAPSS dataset from NASA and place the files as:

```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

---

## Development Principles

- EDA is not optional — understand the data before modeling
- Never train on raw unscaled sensor values
- Remove constant and low-variance features before feature engineering
- Build a working baseline before tuning anything
- Interpretability first, complexity only when justified

---

## Current Status

**Phase 1 — Exploratory Data Analysis (Active Development)**

---

## Author

**Arjit Tripathi**
