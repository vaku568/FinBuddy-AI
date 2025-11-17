import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

def train():
    df = pd.read_csv("finbuddy_master_features.csv", index_col=0)

    X = df.drop(columns=['investment_amount', 'user_archetype'], errors='ignore')
    y = df['investment_amount']

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    print(f"Savings Potential RMSE: {rmse:.4f}")
    joblib.dump(model, "models/savings_potential.pkl")

if __name__ == "__main__":
    train()
