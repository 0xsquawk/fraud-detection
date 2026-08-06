# Credit Card Fraud Detection: An Imbalanced Classification Pipeline

A structured machine learning workflow for credit card fraud detection, encompassing data cleaning, exploratory data analysis, feature engineering, cross validation, and hyperparameter tuning.

**Repository:** https://github.com/0xsquawk/fraud-detection

**Author:** Charchit ([@0xSquawk](https://github.com/0xsquawk))

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Cleaning & Preprocessing](#1-data-cleaning--preprocessing)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Time-Based Train/Test Split](#4-time-based-traintest-split)
  - [5. Model Selection & Cross Validation](#5-model-selection--cross-validation)
  - [6. Hyperparameter Tuning](#6-hyperparameter-tuning)
  - [7. Final Model Training & Evaluation](#7-final-model-training--evaluation)
  - [8. Threshold Tuning](#8-threshold-tuning)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Key Learnings & Design Decisions](#key-learnings--design-decisions)
- [Future Improvements](#future-improvements)
- [Revision History](#revision-history)
- [License](#license)

---

## Overview

This project builds an end-to-end binary classification pipeline to detect fraudulent credit card transactions. The dataset is heavily imbalanced (0.52% fraud), which shapes nearly every decision in the pipeline - from the choice of evaluation metric (PR-AUC over accuracy) to the modeling approach (tree-based ensembles with class-weighting) to the validation strategy (a chronological split instead of a random one, to avoid leaking future information into time-based features).

The full workflow - cleaning, EDA, feature engineering, cross validation, and hyperparameter tuning - is contained in a single Jupyter notebook: [`fraud_analytics.ipynb`](./fraud_analytics.ipynb).

## Problem Statement

Out of 1,852,394 merged transactions, only 9,651 (0.52%) are fraudulent. At this imbalance:

- **Accuracy is not a usable metric** - a model that predicts "not fraud" every time would still be ~99.5% accurate.
- **Raw fields carry no signal on their own.** Timestamps, dates of birth, and merchant/customer coordinates need to be transformed into behavioral and velocity-based features before they're useful to a model.
- **Standard random splitting leaks information.** Several of the engineered features (rolling counts, spend deviation, transaction velocity) are time-dependent, so the train/test split has to respect chronological order.

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection (kartik2112/fraud-detection)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Files used:** `fraudTrain.csv` (1,296,675 rows), `fraudTest.csv` (555,719 rows)
- **Merged size:** 1,852,394 rows × 23 columns (24 after adding a `source` flag)
- **Target variable:** `is_fraud` (binary; 0.52% positive rate)
- **Key raw fields:** transaction amount (`amt`), category, merchant, cardholder demographics (`dob`, `gender`, `job`), transaction and merchant geolocation, and timestamp (`trans_date_trans_time`)

The train and test files are merged for shared preprocessing/feature engineering but retain a `source` column so the original, time-respecting split can be recovered before modeling - a random `train_test_split` would leak future information into the rolling/velocity features built later in the pipeline.

## Project Structure

```
.
├── fraud_analytics.ipynb   # Main notebook: cleaning, EDA, feature engineering, modeling
├── requirements.txt        # Python dependencies
└── README.md
```

## Methodology

### 1. Data Cleaning & Preprocessing

- Loaded `fraudTrain.csv` and `fraudTest.csv` via `kagglehub`, tagged each with a `source` column, and concatenated them into a single working frame.
- Dropped the redundant `Unnamed: 0` index column.
- Cast `trans_date_trans_time` and `dob` to proper `datetime` types.
- Verified data quality: **zero null values** and **zero duplicate rows** across the full merged dataset.
- Normalized the `merchant` column by stripping the `fraud_` prefix present in the raw data.

### 2. Exploratory Data Analysis

**Univariate Analysis**
- Measured skewness and kurtosis across numeric columns - `amt` is extremely right-skewed (skew ≈ 40.8, kurtosis ≈ 4182), driven by a small number of very large transactions.
- Ran a Shapiro-Wilk normality test, confirming `amt` is not normally distributed - supporting the use of median/IQR-based outlier logic over mean/std-based rules.
- Visualized the top categorical values (merchant, category, gender, city, state, job) among fraudulent transactions.
- Compared IQR-based outlier bounds for fraud vs. non-fraud transactions - non-fraud upper bound is ~$119 vs. ~$1,234 for fraud, showing that "large" means something very different depending on the class.

**Bivariate / Multivariate Analysis**
- Compared amount distributions by class: fraud transactions average $530.66 vs. $67.65 for non-fraud (~7.8x higher), with a similarly wide gap in the median ($390 vs. $47.24).
- Measured fraud rate by transaction amount quartile, by category, and by merchant (top 15).
- Aggregated total fraud amount by category to identify where fraud losses concentrate.
- Reviewed a correlation matrix across numeric fields as a baseline sanity check, while noting that linear correlation understates the real relationship given how skewed `amt` is and how rare fraud is - the category/merchant breakdowns and engineered behavioral features capture the non-linear signal instead.

### 3. Feature Engineering

**Transaction-Level Features**
- Extracted `hour`, `day_of_week`, and `is_weekend` from the transaction timestamp.
- Confirmed that fraud counts vary meaningfully by hour and day of week, justifying these as model features.

**Behavioral / Velocity Features**
- Rolling transaction counts per card over 1-hour and 24-hour windows (`tx_count_1h`, `tx_count_24h`).
- 30-day rolling mean and standard deviation of spend per card, used to compute a `spend_z_score` - how anomalous a transaction is relative to that card's own recent history.
- `is_new_history` flag for cards with no trailing 30-day window (first-time or early transactions), since these can't be scored against a "normal" baseline the same way. Rows with no history get a neutral z-score of 0 rather than a misleading skewed value.
- `distance_km` - haversine distance between the cardholder's location and the merchant's location.
- `age` - derived from transaction date and date of birth.
- `txns_rolling_count` - cumulative transaction count per card (a simple velocity signal).
- Kept `merchant`, `job`, and `city` as high-cardinality categoricals (rather than dropping them) since the EDA showed real fraud-rate variation across these groups - they're frequency-encoded further down instead of one-hot encoded, to avoid exploding the feature space.

### 4. Time-Based Train/Test Split

- Sorted the full dataset chronologically and split 70/30 using the original `source` flag, rather than a random split.
- Verified the split holds: training data spans 2019-01-01 to 2020-06-21; test data picks up immediately after, from 2020-06-21 to 2020-12-31, with no overlap.
- Dropped identifier columns (`first`, `last`, `trans_num`, `street`, `dob`, `unix_time`, `cc_num`, `zip`) to avoid training on noise or leakage-prone fields.
- Frequency-encoded `merchant`, `job`, and `city` using **train-only** statistics, applied to both train and test, to avoid target leakage.

### 5. Model Selection & Cross Validation

- Built a 100k-row stratified sample from the training set for fast iteration, preserving the original class imbalance (~1.0% fraud in the sample vs. 0.52% overall).
- Computed `scale_pos_weight` for imbalance-aware boosting.
- Defined a shared preprocessing pipeline (`OrdinalEncoder` + `ColumnTransformer`) for tree-based models.
- Ran an initial cross-validation pass across Random Forest, XGBoost, and Decision Tree using `TimeSeriesSplit` (5 folds) to respect temporal ordering during validation.

### 6. Hyperparameter Tuning

- Replaced the fixed-hyperparameter CV pass with `RandomizedSearchCV` across all three candidate models, each with its own search space (tree depth, learning rate, subsampling ratios, regularization, etc.).
- Scored on **PR-AUC** (average precision), the appropriate metric for a ~0.5% positive class.
- Results on the held-out CV folds:

  | Model | Best PR-AUC | Time |
  |---|---|---|
  | **XGBoost** | **0.8243** | 15.4s |
  | Random Forest | 0.7744 | 76.5s |
  | Decision Tree | 0.5712 | 3.1s |

  XGBoost was selected as the final model - best PR-AUC and fastest search time.

### 7. Final Model Training & Evaluation

- Cloned the best XGBoost pipeline from the search (carrying over its tuned hyperparameters) and refit it on the **full** training set (rather than the 100k-row search sample).
- Recomputed `scale_pos_weight` on the full training distribution.
- Evaluated on the held-out, chronologically later test set.

### 8. Threshold Tuning

- Computed precision and recall across the full range of classification thresholds from the precision-recall curve.
- Searched for a threshold where precision > 0.85 **and** recall > 0.90 simultaneously - none exists in this model/feature configuration.
- Fell back to the default 0.5 threshold, which reproduces the same precision/recall as the untuned prediction.

## Results

**Final XGBoost model - Test Set:**

| Metric | Score |
|---|---|
| PR-AUC | **0.9239** |
| Accuracy | 0.9990 |
| Precision (fraud class) | 0.89 |
| Recall (fraud class) | 0.85 |
| F1-score (fraud class) | 0.87 |

**Confusion Matrix (test set, threshold = 0.5):**

|  | Predicted: Not Fraud | Predicted: Fraud |
|---|---|---|
| **Actual: Not Fraud** | 553,359 | 215 |
| **Actual: Fraud** | 320 | 1,825 |

Out of 555,719 test transactions and 2,145 true fraud cases, the model correctly identifies 1,825 (85%) while flagging only 215 legitimate transactions as false positives.

> Accuracy is reported for completeness but is not a meaningful metric at a 0.52% base fraud rate - PR-AUC and the precision/recall breakdown on the fraud class are the metrics that matter here.

## Feature Importance

Top 10 features by XGBoost feature importance:

| Feature | Importance |
|---|---|
| `city_pop` | 0.36 |
| `spend_z_score` | 0.14 |
| `hour` | 0.09 |
| `category` | 0.09 |
| `rolling_std` | 0.07 |
| `merchant_freq` | 0.06 |
| `rolling_mean` | 0.04 |
| `is_new_history` | 0.04 |
| `tx_count_1h` | 0.03 |
| `age` | 0.02 |

The engineered behavioral features (`spend_z_score`, `rolling_std`, `rolling_mean`, `is_new_history`, `tx_count_1h`) account for a substantial share of total importance, validating the feature engineering step. Raw identifiers were excluded from training entirely.

## Tech Stack

- **Language:** Python 3.12
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Statistics:** scipy
- **Modeling:** scikit-learn (Random Forest, Decision Tree, pipelines, `RandomizedSearchCV`, `TimeSeriesSplit`), XGBoost
- **Geospatial:** haversine
- **Data source:** kagglehub

## Getting Started

### Prerequisites

- Python 3.12+
- A [Kaggle account](https://www.kaggle.com/) configured for `kagglehub` dataset downloads

### Installation

```bash
git clone https://github.com/0xsquawk/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

### Usage

Open and run `fraud_analytics.ipynb` top to bottom in Jupyter or JupyterLab:

```bash
jupyter notebook fraud_analytics.ipynb
```

The dataset downloads automatically via `kagglehub` on the first data-loading cell - no manual download required.

## Key Learnings & Design Decisions

- **PR-AUC over accuracy/ROC-AUC.** At a 0.52% fraud rate, accuracy is misleading and ROC-AUC can look deceptively strong even when precision on the minority class is poor. PR-AUC was used throughout for both model selection and final evaluation.
- **Chronological split, not random.** Several features (rolling counts, spend z-score, transaction velocity) are inherently time-dependent. A random split would let future transactions leak into features computed for earlier ones, inflating validation performance in a way that wouldn't hold up in production.
- **Train-only frequency encoding.** High-cardinality categoricals (`merchant`, `job`, `city`) are encoded using statistics computed only from the training set, then applied to test, to avoid target leakage.
- **Explicit no-history handling.** Rather than letting missing rolling statistics silently bias the model, transactions with no trailing history get a dedicated `is_new_history` flag and a neutral (zero) z-score.

## Future Improvements

- Explore SMOTE or other resampling techniques as an alternative/complement to class-weighting.
- Expand hyperparameter search iterations (`n_iter`) beyond the current budget for a more exhaustive tuning pass.
- Investigate additional velocity features (e.g., time-since-last-transaction, merchant-level velocity) to help close the precision/recall gap identified during threshold tuning.
- Package the final pipeline for batch or real-time scoring outside the notebook environment.

## Revision History

| Date | Changes |
|---|---|
| 2026-07-30 | Added precision-recall threshold search; no threshold satisfies precision > 0.85 and recall > 0.90 together, fell back to default 0.5 threshold |
| 2026-07-29 | Refit tuned XGBoost pipeline on full training set, generated final classification report, confusion matrix and PR-AUC on the held-out test set, pulled feature importances |
| 2026-07-28 | Replaced the fixed-hyperparameter CV pass with `RandomizedSearchCV` across Random Forest, XGBoost and Decision Tree; XGBoost selected as best performer |
| 2026-07-27 | First cross-validation pass (Random Forest / XGBoost / Decision Tree) using `TimeSeriesSplit` on a 100k-row stratified sample; merged development into main |
| 2026-07-26 | Finalized chronological 70/30 train/test split, dropped identifier columns, frequency-encoded `merchant`/`job`/`city` using train-only statistics; merged development into main |
| 2026-07-25 | Added behavioral/velocity features - rolling 1h/24h transaction counts, 30-day rolling mean/std and spend z-score, `is_new_history` flag, haversine `distance_km`, `age`, cumulative `txns_rolling_count` |
| 2026-07-24 | Extracted `hour`, `day_of_week`, `is_weekend` from the transaction timestamp and analyzed fraud rate by hour/day; merged development into main |
| 2026-07-23 | Added bivariate/multivariate analysis - amount distributions by class, fraud rate by amount quartile, fraud rate by category/merchant, correlation matrix |
| 2026-07-22 | Added univariate analysis - skewness/kurtosis, Shapiro-Wilk normality test, pie charts of top categorical values among fraud transactions, IQR/boxplot outlier comparison |
| 2026-07-21 | Initial notebook setup - loaded `fraudTrain.csv`/`fraudTest.csv` via kagglehub, merged into a single frame with a `source` flag, dropped `Unnamed: 0`, cast date columns, checked nulls/duplicates, stripped `fraud_` prefix from `merchant` |

## License

This project is licensed under the MIT License. See the LICENSE file for details.
