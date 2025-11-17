import os
import joblib

MODEL_DIR = "models"

# List model files to process - adjust if needed
model_files = {
    "future_spending.pkl": "future_spending_features.pkl",
    "cashflow_liquidity.pkl": "cashflow_liquidity_features.pkl",
    "life_event_detection.pkl": "life_event_detection_features.pkl",
    "subscription_churn.pkl": "subscription_churn_features.pkl",
    "investment_clustering.pkl": "investment_clustering_features.pkl",
    "seasonal_spending.pkl": "seasonal_spending_features.pkl"
}

for model_file, feature_file in model_files.items():
    model_path = os.path.join(MODEL_DIR, model_file)
    feature_path = os.path.join(MODEL_DIR, feature_file)

    if not os.path.exists(model_path):
        print(f"Model file {model_file} not found. Skipping.")
        continue

    try:
        model = joblib.load(model_path)

        # Try to get feature columns
        if hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_.tolist()
        elif hasattr(model, 'feature_names'):
            feature_cols = model.feature_names
        else:
            # As fallback, ask user to provide manually or skip
            print(f"Model {model_file} missing feature names attribute, skipping.")
            continue

        joblib.dump(feature_cols, feature_path)
        print(f"Extracted and saved feature list for {model_file} to {feature_file}")

    except Exception as e:
        print(f"Failed to load or process {model_file}: {type(e).__name__}: {e}")
