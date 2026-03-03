import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ - Credit Risk Scoring",
    page_icon="💳",
    layout="wide"
)

# ── Load Model & Data ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "best_model.pkl")

@st.cache_data
def load_reference_data():
    data = pd.read_csv(BASE_DIR / "data" / "processed" / "processed_sample.csv")
    data.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in data.columns]
    return data

model = load_model()
data = load_reference_data()
feature_columns = [col for col in data.columns if col not in ['TARGET', 'SK_ID_CURR']]

# ── Helper Functions ──────────────────────────────────────────
def get_risk_band(probability):
    if probability < 0.2:
        return "Very Low Risk", "green"
    elif probability < 0.4:
        return "Low Risk", "lightgreen"
    elif probability < 0.6:
        return "Medium Risk", "orange"
    elif probability < 0.8:
        return "High Risk", "red"
    else:
        return "Very High Risk", "darkred"

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank-card-front-side.png", width=80)
st.sidebar.title("CreditIQ")
st.sidebar.markdown("Credit Risk Scoring System")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Navigate", 
    ["🏠 Home", "🔍 Risk Assessment", "📊 Model Insights", "📈 Data Overview"])

# ── Home Page ─────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("💳 CreditIQ — Credit Risk Scoring System")
    st.markdown("""
    > *Predicting loan default risk using Machine Learning*
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC Score", "0.7522", "✓ Good")
    col2.metric("Gini Coefficient", "0.5044", "✓ Good")
    col3.metric("KS Statistic", "0.3733", "✓ Good")
    col4.metric("Training Samples", "307,511", "")

    st.markdown("---")
    st.subheader("What is CreditIQ?")
    st.markdown("""
    CreditIQ is an end-to-end machine learning system that predicts the 
    probability of a borrower defaulting on a loan. It is inspired by 
    credit risk systems used at:
    
    - 🏦 **Banks** — Equity Bank, KCB, NCBA
    - 📱 **Fintechs** — Tala, Branch, Lipa Later
    - 🌍 **Global lenders** — using the Home Credit dataset
    
    ### How it works
    1. Enter borrower details in **Risk Assessment**
    2. The model predicts default probability
    3. A risk band is assigned (Very Low → Very High)
    4. SHAP values explain *why* the decision was made
    """)

    st.markdown("---")
    st.subheader("Tech Stack")
    col1, col2, col3 = st.columns(3)
    col1.info("**ML Models**\nLightGBM, XGBoost\nRandom Forest\nLogistic Regression")
    col2.info("**Libraries**\nScikit-learn, SHAP\nPandas, NumPy\nimbalanced-learn")
    col3.info("**Deployment**\nStreamlit\nPython 3.11\nJoblib")

# ── Risk Assessment Page ──────────────────────────────────────
elif page == "🔍 Risk Assessment":
    st.title("🔍 Borrower Risk Assessment")
    st.markdown("Enter borrower details to get a default risk prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        age = st.slider("Age", 18, 70, 35)
        gender = st.selectbox("Gender", ["M", "F"])
        education = st.selectbox("Education", [
            "Higher education", "Secondary / secondary special",
            "Incomplete higher", "Lower secondary", "Academic degree"
        ])
        family_members = st.number_input("Family Members", 1, 10, 2)
        owns_car = st.selectbox("Owns Car", ["Y", "N"])
        owns_realty = st.selectbox("Owns Property", ["Y", "N"])

    with col2:
        st.subheader("Financial Info")
        income = st.number_input("Annual Income (KES)", 
                                  10000, 10000000, 180000, step=10000)
        credit_amount = st.number_input("Loan Amount (KES)", 
                                         10000, 5000000, 500000, step=10000)
        annuity = st.number_input("Monthly Annuity (KES)", 
                                   1000, 200000, 25000, step=1000)
        income_type = st.selectbox("Income Type", [
            "Working", "Commercial associate", "Pensioner",
            "State servant", "Unemployed", "Student", "Businessman"
        ])

    with col3:
        st.subheader("Credit History")
        ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.5)
        ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5)
        ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5)
        days_employed = st.number_input("Days Employed", 0, 20000, 1000)
        bureau_loans = st.number_input("Previous Bureau Loans", 0, 50, 3)
        active_loans = st.number_input("Active Loans", 0, 20, 1)

    st.markdown("---")

    if st.button("🔍 Assess Risk", use_container_width=True):
        input_dict = {
            'DAYS_BIRTH': age * -365,
            'CODE_GENDER': gender,
            'NAME_EDUCATION_TYPE': education,
            'CNT_FAM_MEMBERS': family_members,
            'FLAG_OWN_CAR': owns_car,
            'FLAG_OWN_REALTY': owns_realty,
            'AMT_INCOME_TOTAL': income,
            'AMT_CREDIT': credit_amount,
            'AMT_ANNUITY': annuity,
            'NAME_INCOME_TYPE': income_type,
            'EXT_SOURCE_1': ext_source_1,
            'EXT_SOURCE_2': ext_source_2,
            'EXT_SOURCE_3': ext_source_3,
            'DAYS_EMPLOYED': days_employed * -1,
            'BUREAU_LOAN_COUNT': bureau_loans,
            'BUREAU_ACTIVE_LOANS': active_loans,
            'CREDIT_INCOME_RATIO': credit_amount / income,
            'ANNUITY_INCOME_RATIO': annuity / income,
            'CREDIT_ANNUITY_RATIO': credit_amount / annuity,
            'INCOME_PER_PERSON': income / family_members,
        }

        with st.spinner("Analyzing borrower profile..."):
            input_df = preprocess_input(input_dict)
            probability = model.predict_proba(input_df)[:, 1][0]
            risk_band, color = get_risk_band(probability)

        st.markdown("---")
        st.subheader("Assessment Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Default Probability", f"{probability*100:.1f}%")
        col2.metric("Risk Band", risk_band)
        col3.metric("Recommendation", 
                    "✓ Approve" if probability < 0.5 else "✗ Review")

        # Risk gauge
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh(["Risk"], [probability], color=color, height=0.4)
        ax.barh(["Risk"], [1 - probability], left=[probability], 
                color='lightgray', height=0.4)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel("Default Probability")
        ax.set_title(f"Risk Score: {probability*100:.1f}% — {risk_band}")
        st.pyplot(fig)

        # Breakdown
        st.markdown("---")
        st.subheader("Key Risk Factors")
        factors = {
            "Credit to Income Ratio": input_dict['CREDIT_INCOME_RATIO'],
            "Annuity to Income Ratio": input_dict['ANNUITY_INCOME_RATIO'],
            "Avg External Credit Score": np.mean([ext_source_1, ext_source_2, ext_source_3]),
            "Active Loans": active_loans,
        }
        factors_df = pd.DataFrame(factors.items(), 
                                   columns=['Factor', 'Value'])
        st.dataframe(factors_df, use_container_width=True)

# ── Model Insights Page ────────────────────────────────────────
elif page == "📊 Model Insights":
    st.title("📊 Model Insights")
    st.markdown("Understanding what drives credit default predictions.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "ROC Curve", "Risk Bands"])

    with tab1:
        st.subheader("SHAP Feature Importance")
        img_path = BASE_DIR / "reports" / "figures" / "shap_feature_importance.png"
        if img_path.exists():
            st.image(str(img_path), use_column_width=True)
        else:
            st.info("Run notebook 05 to generate SHAP plots")

        st.subheader("SHAP Summary Plot")
        img_path2 = BASE_DIR / "reports" / "figures" / "shap_summary_dot.png"
        if img_path2.exists():
            st.image(str(img_path2), use_column_width=True)

    with tab2:
        st.subheader("ROC Curve")
        roc_path = BASE_DIR / "reports" / "figures" / "roc_curve_final.png"
        if roc_path.exists():
            st.image(str(roc_path), use_column_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("ROC-AUC", "0.7522")
        col2.metric("Gini", "0.5044")
        col3.metric("KS Statistic", "0.3733")

    with tab3:
        st.subheader("Default Rate by Risk Band")
        risk_path = BASE_DIR / "reports" / "figures" / "default_rate_by_risk_band.png"
        if risk_path.exists():
            st.image(str(risk_path), use_column_width=True)

        st.subheader("KS Statistic Plot")
        ks_path = BASE_DIR / "reports" / "figures" / "ks_statistic_plot.png"
        if ks_path.exists():
            st.image(str(ks_path), use_column_width=True)

# ── Data Overview Page ────────────────────────────────────────
elif page == "📈 Data Overview":
    st.title("📈 Data Overview")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applications", "307,511")
    col2.metric("Features Used", "195")
    col3.metric("Default Rate", "8.07%")

    st.markdown("---")

    tab1, tab2 = st.tabs(["Target Distribution", "Missing Values"])

    with tab1:
        img = BASE_DIR / "reports" / "figures" / "target_distribution_pie.png"
        if img.exists():
            st.image(str(img), use_column_width=True)

    with tab2:
        img2 = BASE_DIR / "reports" / "figures" / "top20_missing_values.png"
        if img2.exists():
            st.image(str(img2), use_column_width=True)

    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(data.head(10), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Built by Benedict Bett | "
    "<a href='https://github.com/bennedictbett'>GitHub</a> | "
    "<a href='https://bennedictbett.github.io/portfolio-project/'>Portfolio</a>"
    "</center>",
    unsafe_allow_html=True
)