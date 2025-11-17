import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

def train():
    df = pd.read_csv("users_profile_full_v3.csv")

    archetypes = {
        "conservative_saver": 0,
        "balanced_planner": 1,
        "aggressive_investor": 2,
        "impulsive_spender": 3,
        "goal_oriented_optimizer": 4
    }
    df['user_archetype_code'] = df['user_archetype'].map(archetypes)

    df = df.dropna(subset=['user_archetype_code'])
    df['user_archetype_code'] = df['user_archetype_code'].astype(int)

    X = df.drop(columns=['user_id', 'user_archetype', 'user_archetype_code'])
    y = df['user_archetype_code']

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    X.fillna(-1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Archetype Classifier Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=archetypes.keys()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, "models/archetype_classifier.pkl")
    joblib.dump(archetypes, "models/archetype_label_mapping.pkl")

if __name__ == "__main__":
    train()
