import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

def train():
    df = pd.read_csv("finbuddy_master_features.csv", index_col=0)
    expense_cols = ['food_expense','groceries_expense','education_expense','subscriptions_expense',
                    'fuel_expense','transportation_expense','utilities_expense','entertainment_expense',
                    'shopping_expense','healthcare_expense','personal_care_expense','miscellaneous_expense']

    X = df.drop(columns=expense_cols + ['user_archetype'], errors='ignore')

    for cat in expense_cols:
        y = df[cat]
        if y.nunique() < 2:
            continue

        X_cat = X.copy()
        for col in X_cat.select_dtypes(include=['object']).columns:
            X_cat[col] = X_cat[col].astype('category').cat.codes

        X_train, X_test, y_train, y_test = train_test_split(
            X_cat, y, test_size=0.2, random_state=42)

        model = LGBMRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = sqrt(mse)
        print(f"Category Forecast {cat} RMSE: {rmse:.4f}")
        joblib.dump(model, f"models/category_forecast_{cat}.pkl")

if __name__ == "__main__":
    train()
