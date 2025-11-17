import pandas as pd
from sklearn.cluster import KMeans
from joblib import dump

df = pd.read_csv("finbuddy_master_features.csv", index_col=0)
invest_cols = ['stocks', 'sip', 'crypto', 'gold_bonds', 'total_investment_value']
X = df[invest_cols].fillna(0)

model = KMeans(n_clusters=5, random_state=42)
model.fit(X)
dump(model, "models/investment_clustering.pkl")
print("Investment Clustering model trained.")
