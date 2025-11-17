import os
import pickle

MODEL_DIR = "models"
EXPECTED_MODELS = [
    "spend_extra_trees.pkl",
    "spend_random_forest.pkl",
    "spend_linear.pkl",
    "cashflow_liquidity.pkl",
    "lifeevent_classifier.pkl",
    "subscription_churn.pkl",
    "investment_cluster_kmeans.pkl",
    "seasonal_spending_kmeans.pkl"
]

print("====== Verifying required FinBuddy model files ======")
for fname in EXPECTED_MODELS:
    fpath = os.path.join(MODEL_DIR, fname)
    label = fname.replace(".pkl", "")
    if not os.path.exists(fpath):
        print(f"❌ {fname} — NOT FOUND")
        continue
    try:
        with open(fpath, "rb") as f:
            _ = pickle.load(f)
        print(f"✅ {fname} — Load OK")
    except Exception as e:
        print(f"⚠️ {fname} — Unpickling error: {e.__class__.__name__}, {str(e)}")

print("====== Model verification complete ======")
