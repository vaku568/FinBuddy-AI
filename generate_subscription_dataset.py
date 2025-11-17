"""
Generate Synthetic Subscription Dataset for FinBuddy
----------------------------------------------------
Creates a 12-month subscription activity dataset per user.
Fixed: CSV now saves correctly with full path + folder creation.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

# ======================================================
# ✅ PARAMETERS
# ======================================================
NUM_USERS = 10000
MONTHS = 12
START_DATE = datetime(2024, 1, 1)

# Output directory (ALWAYS saves successfully here)
OUTPUT_DIR = "artifacts"
OUT_FILE = os.path.join(OUTPUT_DIR, "subscriptions_12months.csv")

np.random.seed(42)

# ======================================================
# ✅ Create output folder if missing
# ======================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# ✅ Month list
# ======================================================
months = [
    (START_DATE + pd.DateOffset(months=i)).strftime("%Y-%m")
    for i in range(MONTHS)
]

# ======================================================
# ✅ Generate synthetic user IDs
# ======================================================
user_ids = [f"U{str(i).zfill(5)}" for i in range(1, NUM_USERS + 1)]

# ======================================================
# ✅ Simulate subscription activity
# ======================================================
rows = []

for user in user_ids:
    base_subs = np.random.randint(2, 10)
    monthly_fee = np.random.randint(100, 1000)
    churn_risk = np.random.choice([0, 1], p=[0.85, 0.15])

    for i, month in enumerate(months):

        active_subs = max(0, base_subs - np.random.poisson(0.2 * i))
        canceled_subs = np.random.poisson(0.1 * base_subs)

        total_fee = active_subs * monthly_fee
        auto_renew_flag = np.random.choice([0, 1], p=[0.2, 0.8])

        churn_flag = int(
            (active_subs < 2 and canceled_subs > 1)
            or (churn_risk == 1 and np.random.rand() < 0.3)
        )

        # festival spikes
        if month.endswith(("-11", "-12")):
            total_fee *= np.random.uniform(1.1, 1.4)

        rows.append([
            user,
            month,
            active_subs,
            canceled_subs,
            monthly_fee,
            total_fee,
            auto_renew_flag,
            churn_flag
        ])

# ======================================================
# ✅ Build DataFrame
# ======================================================
df = pd.DataFrame(rows, columns=[
    "user_id",
    "month",
    "active_subs",
    "canceled_subs",
    "avg_sub_fee",
    "total_fee_paid",
    "auto_renew_flag",
    "churn_flag"
])

# ======================================================
# ✅ Add extra churn metrics
# ======================================================
df["churn_rate"] = df.groupby("user_id")["churn_flag"].transform(lambda x: x.rolling(3, 1).mean())
df["subs_to_fee_ratio"] = df["active_subs"] / (df["avg_sub_fee"] + 1)

# ======================================================
# ✅ Save CSV with error handling
# ======================================================
try:
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    abs_path = os.path.abspath(OUT_FILE)
    print(f"\n[SAVED SUCCESSFULLY] → {abs_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}\n")
    print(df.head())

except Exception as e:
    print("\n❌ ERROR: CSV could not be saved")
    print(str(e))
