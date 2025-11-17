import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

def train():
    df = pd.read_csv("finbuddy_master_features.csv", index_col=0)

    X = df.drop(columns=['progress_percent', 'user_archetype'], errors='ignore')
    y = df['progress_percent']

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = Lasso()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    print(f"Goal Achievement RMSE: {rmse:.4f}")
    joblib.dump(model, "models/goal_achievement.pkl")

if __name__ == "__main__":
    train()
