import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

def train():
    df = pd.read_csv("finbuddy_master_features.csv", index_col=0)
    target_col = 'merchant_insights'

    if target_col not in df.columns:
        print(f"Column '{target_col}' not found in dataset. Skipping merchant_behavior model training.")
        return

    X = df.drop(columns=[target_col, 'user_archetype'], errors='ignore')
    y = df[target_col]

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    print(f"Merchant Behavior RMSE: {rmse:.4f}")

    joblib.dump(model, "models/merchant_behavior.pkl")

if __name__ == "__main__":
    train()
