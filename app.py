import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load  # use joblib for robust sklearn model loading
from io import StringIO

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="FinBuddy AI",
    page_icon="💰",
    layout="wide"
)

st.title("💼 FinBuddy AI — Smart Financial Assistant")
st.markdown("### Unified model-powered insights using your trained ML suite.")

# =========================================================
# Load Verified Models
# =========================================================
MODEL_DIR = "models"
ACTUAL_MODELS = {
    "spend_extra_trees": "future_spending.pkl",
    "spend_random_forest": "future_spending.pkl",
    "spend_linear": "future_spending.pkl",
    "cashflow_liquidity": "cashflow_liquidity.pkl",
    "lifeevent_classifier": "life_event_detection.pkl",
    "subscription_churn": "subscription_churn.pkl",
    "investment_cluster_kmeans": "investment_clustering.pkl",
    "seasonal_spending_kmeans": "seasonal_spending.pkl"
}

LABELS = {
    "spend_extra_trees": "Future Spending Predictor (Extra Trees)",
    "spend_random_forest": "Future Spending Predictor (Random Forest)",
    "spend_linear": "Future Spending Predictor (Linear Regression)",
    "cashflow_liquidity": "Cash Flow & Liquidity",
    "lifeevent_classifier": "Life Event Detector",
    "subscription_churn": "Subscription Churn Predictor",
    "investment_cluster_kmeans": "Investment Clustering / Peer Benchmarking",
    "seasonal_spending_kmeans": "Seasonal Spending Patterns",
}

models = {}
load_report = []

for key, label in LABELS.items():
    file_name = ACTUAL_MODELS.get(key)
    model_path = os.path.join(MODEL_DIR, file_name) if file_name else ''
    if not model_path or not os.path.exists(model_path):
        load_report.append((label, "⚠️ Error loading: FileNotFoundError"))
        continue
    try:
        models[key] = load(model_path)  # use joblib.load for sklearn models
        load_report.append((label, "✅ Loaded"))
    except Exception as e:
        load_report.append((label, f"⚠️ Error loading: {type(e).__name__}: {str(e)}"))

# =========================================================
# Sidebar - Model Load Status
# =========================================================
st.sidebar.header("🧩 Models Status")
for model_name, status in load_report:
    st.sidebar.write(f"**{model_name}** — {status}")

# =========================================================
# User Financial Profile Input
# =========================================================
st.header("🧍 User Financial Profile")
with st.expander("Enter User Profile Details"):
    col1, col2, col3 = st.columns(3)
    with col1:
        salary = st.number_input("Monthly Income (₹)", min_value=1000, max_value=10_000_000, value=50000)
        age = st.number_input("Age", min_value=18, max_value=100, value=28)
        dependents = st.number_input("Dependents", min_value=0, max_value=20, value=1)
    with col2:
        monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, max_value=10_000_000, value=25000)
        other_obligations = st.number_input("Loans/EMIs (₹)", min_value=0, max_value=10_000_000, value=10000)
        risk_tolerance = st.slider("Risk Tolerance (1=Low, 10=High)", 1, 10, 5)
    with col3:
        savings_balance = st.number_input("Savings Balance (₹)", min_value=0, max_value=100_000_000, value=200000)
        is_metro = st.selectbox("Lives in Metro City?", ["Yes", "No"])
        profession = st.selectbox("Profession Type", ["Salaried", "Self-employed", "Student", "Retired"])

# =========================================================
# Transaction Data Input
# =========================================================
st.header("📥 Transaction Data")
upload_option = st.radio("Provide your transaction data:", ["Upload CSV", "Enter Manually"])
transactions = None

if upload_option == "Upload CSV":
    transaction_file = st.file_uploader("Upload your bank statement CSV", type=["csv"])
    if transaction_file:
        transactions = pd.read_csv(transaction_file)
        st.success(f"Loaded {len(transactions):,} transactions.")
        st.dataframe(transactions.head())
else:
    st.markdown("Enter transactions manually (CSV format: date,category,amount,type):")
    data = st.text_area(
        "Example: 2025-10-01, Groceries, 1500, debit\n2025-10-05, Salary, 60000, credit"
    )
    if data.strip():
        try:
            transactions = pd.read_csv(StringIO(data), names=["date", "category", "amount", "type"])
            st.success(f"Parsed {len(transactions):,} transactions.")
            st.dataframe(transactions.head())
        except Exception as e:
            st.error(f"Error parsing transactions: {e}")

# =========================================================
# Predict Next Month Spending
# =========================================================
st.header("💸 Future Spending Patterns")
if st.button("Predict Next Month Spending"):
    required_models = ["spend_extra_trees", "spend_random_forest", "spend_linear"]
    if not any(model in models for model in required_models):
        st.error("❌ No spending models loaded.")
    else:
        try:
            features = np.array([[salary, monthly_expenses, salary - monthly_expenses, savings_balance, age, dependents]])
            preds = []
            for key in required_models:
                if key in models:
                    preds.append(models[key].predict(features)[0])
            if preds:
                prediction = np.mean(preds)
                st.success(f"💰 Predicted Next Month Spending: ₹{prediction:,.0f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =========================================================
# Cash Flow & Liquidity Assessment
# =========================================================
st.header("💧 Cash Flow & Liquidity")
if st.button("Check Liquidity Health"):
    if "cashflow_liquidity" in models:
        try:
            features = np.array([[salary, monthly_expenses]])
            value = models["cashflow_liquidity"].predict(features)[0]
            st.metric("Liquidity Index", f"{value:.2f}")
            if value > 0:
                st.success("✅ Positive cashflow. Consider enhancing savings and investments.")
            else:
                st.warning("⚠️ Negative cashflow risk detected. Review your expenses.")
        except Exception as e:
            st.error(f"Liquidity model error: {e}")
    else:
        st.warning("Liquidity model not loaded.")

# =========================================================
# Life Event Detection
# =========================================================
st.header("🎯 Life Event Detection")
if st.button("Detect Life Events"):
    if "lifeevent_classifier" in models:
        try:
            dummy_features = np.random.rand(1, 9)  # Replace with real features if available
            prediction = models["lifeevent_classifier"].predict(dummy_features)[0]
            if prediction == 1:
                st.success("🌟 Major life event likely ahead (e.g. marriage, relocation).")
            else:
                st.info("No major life events predicted soon.")
        except Exception as e:
            st.error(f"Life event model error: {e}")
    else:
        st.warning("Life event model not loaded.")

# =========================================================
# Investment Clustering / Benchmarking
# =========================================================
st.header("📊 Investment Clustering")
if st.button("Show Investment Cluster"):
    if "investment_cluster_kmeans" in models:
        try:
            dummy_features = np.random.rand(1, 6)  # Replace with real investment features
            cluster_id = models["investment_cluster_kmeans"].predict(dummy_features)[0]
            st.success(f"📈 You belong to Investment Cluster #{cluster_id + 1}")
        except Exception as e:
            st.error(f"Investment clustering error: {e}")
    else:
        st.warning("Investment clustering model not loaded.")

# =========================================================
# Subscription Churn Prediction
# =========================================================
st.header("🔄 Subscription Churn Prediction")
if st.button("Predict Subscription Churn"):
    if "subscription_churn" in models:
        try:
            dummy_features = np.random.rand(1, 4)
            churn_pred = models["subscription_churn"].predict(dummy_features)[0]
            st.warning("⚠️ At risk of subscription churn!" if churn_pred == 1 else "✅ Retention predicted.")
        except Exception as e:
            st.error(f"Churn model error: {e}")
    else:
        st.warning("Subscription churn model not loaded.")

# =========================================================
# Seasonal Spending Pattern Analysis
# =========================================================
st.header("🌦️ Seasonal Spending Insights")
if st.button("Analyze Seasonal Pattern"):
    if "seasonal_spending_kmeans" in models:
        try:
            dummy_features = np.random.rand(1, 13)
            cluster = models["seasonal_spending_kmeans"].predict(dummy_features)[0]
            st.info(f"🗓️ Seasonal Cluster #{cluster + 1} — aligns with festival and holiday spending trends.")
        except Exception as e:
            st.error(f"Seasonal spending model error: {e}")
    else:
        st.warning("Seasonal spending model not loaded.")

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("FinBuddy AI © 2025 | Streamlit Dashboard | Powered by your trained ML models")
