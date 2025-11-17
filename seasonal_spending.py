import pandas as pd
from sklearn.cluster import KMeans
from joblib import dump

df = pd.read_csv("finbuddy_master_features.csv", index_col=0)
X = df[['seasonality_index']].fillna(0)

model = KMeans(n_clusters=4, random_state=42)
model.fit(X)
dump(model, "models/seasonal_spending.pkl")
print("Seasonal Spending Clustering trained.")
