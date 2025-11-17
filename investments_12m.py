import pandas as pd
import numpy as np
from tqdm import tqdm

# ===== File Paths =====
USER_PROFILE_FILE = "users_profile_full_v3.csv"
OUTPUT_FILE = "investment_data_12months.csv"

MONTHS = [
    "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10",
    "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04"
]

np.random.seed(42)

# Archetype discipline mapping (0–1 scale)
ARCHETYPE_DISCIPLINE = {
    "Planner": 0.9,
    "Saver": 0.85,
    "Balanced": 0.7,
    "Spender": 0.4,
    "Risky": 0.5,
    "Stressed": 0.3
}

# Risk allocation logic
RISK_ALLOCATION = {
    "Low":    {"stocks": 0.20, "sip": 0.70, "crypto": 0.00, "gold_bonds": 0.10},
    "Medium": {"stocks": 0.45, "sip": 0.45, "crypto": 0.07, "gold_bonds": 0.03},
    "High":   {"stocks": 0.60, "sip": 0.25, "crypto": 0.12, "gold_bonds": 0.03},
}

def simulate_investment_growth(amount):
    monthly_return = np.random.normal(0.01, 0.03)
    return max(0, amount * (1 + monthly_return))

def generate_investment_data():
    df = pd.read_csv(USER_PROFILE_FILE)

    investment_records = []

    print("\n==========================================")
    print(" ✅ Generating Investment & Savings Dataset")
    print("==========================================\n")

    for _, user in tqdm(df.iterrows(), total=len(df)):
        user_id = user["user_id"]
        surplus = max(0, user["monthly_surplus"])
        risk = user["risk_tolerance"]
        archetype = user["user_archetype"]

        discipline = ARCHETYPE_DISCIPLINE.get(archetype, 0.6)
        risk_alloc = RISK_ALLOCATION.get(risk, RISK_ALLOCATION["Medium"])

        invest_base = surplus * np.random.uniform(0.20, 0.80) * discipline

        # track growing investments
        stock_value = gold_bond_value = crypto_value = sip_value = 0

        for month in MONTHS:
            skip_prob = np.random.uniform(0.05, 0.30)

            if np.random.rand() < skip_prob:
                month_invest = invest_base * np.random.uniform(0.0, 0.4)
            else:
                month_invest = invest_base

            stocks = month_invest * risk_alloc["stocks"]
            sip = month_invest * risk_alloc["sip"]
            crypto = month_invest * risk_alloc["crypto"]
            gold_bonds = month_invest * risk_alloc["gold_bonds"]

            # Apply growth over months
            stock_value = simulate_investment_growth(stock_value + stocks)
            sip_value = simulate_investment_growth(sip_value + sip)
            crypto_value = simulate_investment_growth(crypto_value + crypto)
            gold_bond_value = simulate_investment_growth(gold_bond_value + gold_bonds)

            investment_records.append({
                "user_id": user_id,
                "month": month,
                "investment_monthly": round(month_invest, 2),
                "stocks": round(stocks, 2),
                "sip": round(sip, 2),
                "crypto": round(crypto, 2),
                "gold_bonds": round(gold_bonds, 2),
                "skip_month": int(month_invest < invest_base * 0.5),
                "total_investment_value": round(
                    stock_value + sip_value + crypto_value + gold_bond_value, 2
                )
            })

    df_out = pd.DataFrame(investment_records)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Dataset Created Successfully!")
    print(f"Total rows: {len(df_out)}")
    print(f"Saved file: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    generate_investment_data()
