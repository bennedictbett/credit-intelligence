# Credit-Intelligence — Credit Risk Scoring System

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red) ![License](https://img.shields.io/badge/license-MIT-green) ![Model](https://img.shields.io/badge/Model-LightGBM-orange)

> **Predicting loan default risk using Machine Learning** — an end-to-end credit scoring system inspired by real-world lenders in Kenya and globally.

🔗 **Live App:** [credit-intelligence-53xbnpkgxumwu3w7jycx85.streamlit.app](https://credit-intelligence-53xbnpkgxumwu3w7jycx85.streamlit.app)

---

## Overview

CreditIQ is a machine learning system that predicts the probability of a borrower defaulting on a loan. It is inspired by credit risk systems used by:

- 🏦 **Banks** — Equity Bank, KCB, NCBA
- 📱 **Fintechs** — Tala, Branch, Lipa Later
- 🌍 **Global lenders** — using the [Home Credit dataset](https://www.kaggle.com/c/home-credit-default-risk)

The system takes borrower information as input, runs it through a trained ML model, assigns a risk band, and explains the decision using SHAP values.

---

## Features

- **Risk Assessment Form** — Enter borrower details across Personal Info, Financial Info, and Credit History
- **Default Probability Score** — Model outputs a calibrated probability (0–100%)
- **Risk Band Classification** — Very Low / Low / Medium / High / Very High
- **Loan Recommendation** — Approve / Review / Decline
- **SHAP Explainability** — Visual explanation of which features drove the decision
- **Model Performance Dashboard** — ROC-AUC, Gini Coefficient, KS Statistic

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.7522 |
| Gini Coefficient | 0.5044 |
| KS Statistic | 0.3733 |
| Training Samples | 307,511 |

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML Models | LightGBM, XGBoost, Random Forest, Logistic Regression |
| Feature Engineering | Pandas, NumPy, Scikit-learn |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit Cloud |
| Language | Python 3.11 |

---

## Project Structure

```
credit-intelligence/
├── app/
│   ├── main.py                  # Streamlit app entry point
│   ├── pages/                   # Multi-page app pages
│   └── components/              # Reusable UI components
├── src/
│   ├── data_processing.py       # Feature engineering pipeline
│   └── train.py                 # Model training script
├── data/
│   └── processed/
│       └── processed_sample.csv # Sample dataset (61,502 rows)
├── models/
│   └── best_model.pkl           # Trained LightGBM model
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Enter borrower details** in the Risk Assessment form
2. **The model predicts** default probability using a trained LightGBM classifier
3. **A risk band is assigned** (Very Low → Very High) based on the probability score
4. **SHAP values explain** why the decision was made — which features pushed the score up or down

---

## Dataset

- **Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
- **Size:** 307,511 training samples, 122 features
- **Target:** Binary — 1 (default) / 0 (no default)
- **Class imbalance:** ~8% positive (default) rate, handled with SMOTE resampling

---

## Known Limitations & Future Work

- **Class imbalance** — The dataset has ~8% default rate, causing the model to skew toward approvals. Threshold tuning and improved resampling strategies would improve real-world recall.
- **Feature scope** — The current model uses a subset of available features. Incorporating bureau and installment data would likely improve AUC further.
- **Localisation** — Future versions could incorporate Kenya-specific bureau data (CRB Africa, Metropol) for more relevant predictions.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/bennedictbett/credit-intelligence.git
cd credit-intelligence

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

---

## Author

**Benedict Bett**
- GitHub: [@bennedictbett](https://github.com/bennedictbett)
- App: [CreditIQ on Streamlit](https://credit-intelligence-53xbnpkgxumwu3w7jycx85.streamlit.app)

---

*Built as a portfolio project demonstrating end-to-end ML system design, feature engineering, model training, and production deployment.*