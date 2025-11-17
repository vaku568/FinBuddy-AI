import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# ✅ Input: Your previously generated transactions dataset
TRANSACTION_FILE = "transaction_data_12months.csv"
OUTPUT_FILE = "fraud_signals_12months.csv"

# ✅ Fraud Parameters
FRAUD_RATE = 0.035  # 3.5%
FRAUD_TYPES = [
    "Card Skimming",
    "Phishing",
    "Account Takeover",
    "Fake Merchant",
    "Location Mismatch",
    "Unusual High Transaction",
    "Multiple Small Transactions",
]

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

def generate_fraud_signals():
    print("\n📌 Loading transaction dataset...")
    df_tx = pd.read_csv(TRANSACTION_FILE)

    total_transactions = len(df_tx)
    num_fraud = int(total_transactions * FRAUD_RATE)

    print(f"✅ Total transactions: {total_transactions:,}")
    print(f"⚠️ Fraud sample target: {num_fraud:,}")

    fraud_samples = df_tx.sample(num_fraud, random_state=42).copy()
    fraud_samples.reset_index(drop=True, inplace=True)

    fraud_data = []
    for idx, row in tqdm(fraud_samples.iterrows(), total=num_fraud, desc="Generating fraud labels"):

        fraud_type = random.choice(FRAUD_TYPES)
        severity = random.choices(SEVERITY_LEVELS, weights=[0.2, 0.4, 0.3, 0.1], k=1)[0]

        flagged_amount = round(row["amount"] * np.random.uniform(1.1, 2.5), 2)

        fraud_data.append([
            row["transaction_id"], row["user_id"], row["date"],
            fraud_type, severity,
            flagged_amount,
            "Fraudulent"  # ✅ ground truth label
        ])

    df_fraud = pd.DataFrame(
        fraud_data,
        columns=[
            "transaction_id", "user_id", "date",
            "fraud_type", "severity",
            "flagged_amount",
            "fraud_label"
        ]
    )

    df_fraud.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Fraud dataset created successfully!")
    print(f"📄 File saved: {OUTPUT_FILE}")
    print(f"⚠️ Fraud rows: {len(df_fraud):,}")

if __name__ == "__main__":
    generate_fraud_signals()
