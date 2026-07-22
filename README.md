# Credit Card Fraud Detection

An end-to-end machine learning project for detecting fraudulent credit card transactions in a large-scale, class-imbalanced dataset of simulated cardholder transactions.

## Objective

Build an end-to-end fraud detection pipeline that identifies fraudulent credit card transactions in a class-imbalanced, geolocation- and behavior-rich dataset. The project engineers features that capture spending anomalies, merchant risk, and transaction velocity, and benchmarks multiple classifiers under appropriate cross-validation to maximize fraud recall while controlling false positives.

## Dataset

Source: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/kuldeepjangra/credit-card-fraud-detection-500k-transactions)

The dataset contains approximately 1.85 million simulated card transactions spanning 2019-2020, including:

- Transaction details: timestamp, amount, merchant, category
- Cardholder attributes: demographics, date of birth, home location
- Merchant location at time of transaction
- Binary fraud label (`is_fraud`)

The dataset is highly imbalanced, with fraudulent transactions representing under 1 percent of all records.

## Business Questions

**Detection and risk drivers**

- What transaction, behavioral, and demographic patterns most strongly distinguish fraudulent from legitimate transactions?
- Which merchant categories and individual merchants carry disproportionate fraud risk?
- Does the distance between cardholder location and merchant location predict fraud likelihood?
- Are there identifiable time-based patterns in fraud occurrence, such as hour of day or day of week?
- Does cardholder age or spending relative to their historical norm correlate with fraud?

**Operational and cost framing**

- What is the estimated financial exposure prevented at a given model recall level, and what is the tradeoff against false-positive-driven customer friction?
- What classification threshold balances fraud recall against realistic fraud analyst review capacity?
- Which model, evaluated on precision-recall AUC rather than accuracy, performs best under a fraud rate below 1 percent?

**Feature engineering**

- Can transaction velocity features, such as rolling transaction counts or deviation from a cardholder's historical spending average, improve detection beyond point-in-time transaction attributes alone?

## Project Structure

The analysis follows a standard applied ML workflow:

1. Data Preprocessing
2. Exploratory Data Analysis
3. Feature Engineering
4. Feature Scaling/Encoding/Standardization
5. Cross Validation for Model Selection
6. Data Modeling

## Methodology Notes

- Personally identifiable columns (name, street address, transaction number, card number) are excluded or hashed prior to modeling.
- Given the severe class imbalance, evaluation prioritizes precision-recall AUC and recall over accuracy.
- Train/test splitting is performed on a time basis rather than randomly, since individual cardholders appear across many transactions and random splitting would leak cardholder history between sets.
- Cross-validation uses stratified folds to preserve the fraud/non-fraud ratio across splits.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Status

Work in progress. Preprocessing and initial data quality checks are complete; EDA, feature engineering, and modeling are in development.

## Author

Charchit
