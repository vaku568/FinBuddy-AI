import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

df = pd.read_csv("finbuddy_master_features.csv", index_col=0)
y = (df['churn_flag'] > 0.5).astype(int)
X = df.drop(columns=['churn_flag', 'user_archetype'], errors='ignore')

for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Subscription Churn Accuracy: {acc:.4f}")
dump(model, "models/subscription_churn.pkl")
