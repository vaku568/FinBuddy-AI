"""
income_pattern_12m.py
Generates monthly income (May 2024 -> Apr 2025) time-series per user
based on a base monthly_income column inside the user profile CSV.

Outputs:
  - income_data_12months.csv

Notes:
  - This script will look for user profile filenames in order:
      users_profile_full_v3.csv, users_profile_full.csv, user_profile_10k.csv
    and will error if none found.
  - It uses realistic growth + volatility per user archetype / student / experience.
"""

import pandas as pd
import numpy as np
import random
import os

# ---------------- CONFIG ----------------
POSSIBLE_USER_FILES = [
    "users_profile_full_v3.csv",
    "users_profile_full.csv",
    "user_profile_10k.csv",
    "users_profile_full_v3.xlsx",
    "users_profile_full.xlsx"
]
OUTPUT_FILE = "income_data_12months.csv"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Months: May 2024 -> Apr 2025
MONTHS = [
    "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10",
    "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04"
]

# Income types probabilities baseline (adjusted per user)
BASE_INCOME_TYPES = ["salary", "commission", "freelance", "stipend"]
# We'll infer income_type from profile if column exists; otherwise assign probabilistically.

# Helper utilities
def find_user_file():
    for f in POSSIBLE_USER_FILES:
        if os.path.exists(f):
            return f
    raise FileNotFoundError(
        "No user profile file found. Looked for: " + ", ".join(POSSIBLE_USER_FILES)
    )

def safe_get(row, key, default=None):
    return row.get(key) if key in row else default

def choose_income_type(row):
    # If user file includes an 'income_type' column, use it (normalized)
    it = safe_get(row, "income_type", None)
    if pd.notnull(it) and isinstance(it, str):
        it = it.strip().lower()
        if it in BASE_INCOME_TYPES:
            return it
    # otherwise heuristics:
    is_student = str(safe_get(row, "is_student", "No")).strip().lower() in ["yes", "true", "1"]
    age = int(safe_get(row, "age", 30) or 30)
    years_exp = float(safe_get(row, "years_experience", 3) or 0)

    # Students -> stipend with high chance
    if is_student:
        return "stipend" if random.random() < 0.7 else "freelance"

    # Young with low experience -> freelancer or salary
    if years_exp < 2:
        return "freelance" if random.random() < 0.35 else "salary"

    # Metro & higher experience -> salary/commission mix
    is_metro = str(safe_get(row, "is_metro", "No")).strip().lower() in ["yes", "true", "1"]
    if is_metro and years_exp >= 3:
        return "commission" if random.random() < 0.25 else "salary"

    # Default
    return "salary"

def compute_base_growth_rate(years_experience):
    """
    Return an annualized growth rate (decimal) applied progressively.
    More experience -> smaller but steadier raises; junior -> higher % variability.
    """
    if years_experience is None:
        years_experience = 3.0
    try:
        y = float(years_experience)
    except:
        y = 3.0
    if y < 1:
        return 0.05  # 5% annual approx for early-career (but can be variable)
    if y < 4:
        return 0.04
    if y < 8:
        return 0.03
    return 0.02

def month_growth_multiplier(annual_rate):
    """
    Convert annual rate to a modest per-quarter step: we will add growth
    as small increments every 3 months to simulate raises/promotions.
    Return list of length 12 with cumulative multipliers.
    """
    # We'll implement quarterly bump: every 3 months add approx (annual_rate/4)
    quarter_inc = annual_rate / 4.0
    multipliers = []
    cum = 1.0
    for m in range(12):
        if m % 3 == 0 and m != 0:
            cum *= (1 + quarter_inc)
        multipliers.append(cum)
    return multipliers

def simulate_month_income(base_income, income_type, monthly_multiplier, volatility_params):
    """
    Generate an income for a month given a base income and type.
    volatility_params: dict with sigma (std dev fraction), bonus_prob, bonus_scale
    """
    sigma = volatility_params.get("sigma", 0.02)
    # Base amount (may be fraction of base for stipend/freelance)
    if income_type == "stipend":
        base = base_income * 0.6  # stipend usually smaller than base_income
    elif income_type == "freelance":
        base = base_income * 0.7
    elif income_type == "commission":
        base = base_income * 0.9
    else:
        base = base_income

    # Apply growth multiplier (from promotions/raises)
    base = base * monthly_multiplier

    # Normal noise
    noise = np.random.normal(loc=0.0, scale=sigma * base)
    month_amount = base + noise

    # Commission/freelance occasionally get spikes / zeros
    if income_type == "freelance":
        # chance of zero or big spike
        if random.random() < 0.08:
            # lean month
            month_amount *= random.uniform(0.0, 0.6)
        if random.random() < 0.12:
            # big contract
            spike = base * random.uniform(0.8, 3.0)
            month_amount += spike
    if income_type == "commission":
        # occasional bonus spikes
        if random.random() < 0.15:
            month_amount += base * random.uniform(0.2, 1.0)

    # Stipend tends to be stable but sometimes missed
    if income_type == "stipend":
        if random.random() < 0.02:
            month_amount *= random.uniform(0.5, 0.95)

    # Ensure non-negative and round to 2 decimals
    month_amount = max(0.0, round(float(month_amount), 2))
    return month_amount

def compute_volatility_score(monthly_values):
    arr = np.array(monthly_values, dtype=float)
    if arr.mean() <= 0:
        return 100.0
    # volatility measured as coefficient of variation (std / mean) scaled 0-100
    cov = arr.std() / (arr.mean() + 1e-9)
    score = min(100.0, cov * 100.0 * 1.5)  # scale factor to get interpretable 0-100
    return round(float(score), 2)

def compute_income_stability_index(volatility_score, missed_payments_count=0):
    # stability decreases with volatility and missed payments
    penalty = missed_payments_count * 3.0
    idx = max(0.0, 100.0 - volatility_score - penalty)
    return round(float(idx), 2)

# ---------------- Main generation ----------------
def generate_income_dataset():
    user_file = find_user_file()
    print("Loading user profiles from:", user_file)
    if user_file.endswith(".xlsx"):
        df_users = pd.read_excel(user_file)
    else:
        df_users = pd.read_csv(user_file)

    # check required column existence
    if "user_id" not in df_users.columns:
        raise KeyError("User profile file must contain 'user_id' column")
    if "monthly_income" not in df_users.columns:
        # if monthly_income absent, try monthly_income column alternatives
        alternatives = [c for c in df_users.columns if "income" in c.lower()]
        if alternatives:
            df_users = df_users.rename(columns={alternatives[0]: "monthly_income"})
        else:
            raise KeyError("User profile file must contain 'monthly_income' column or similar")

    output_rows = []
    skipped_users = 0

    for idx, urow in df_users.iterrows():
        user = urow.to_dict()
        user_id = user["user_id"]
        try:
            base_income = float(user.get("monthly_income", 0) or 0)
        except:
            base_income = 0.0

        # decide income type
        income_type = choose_income_type(user)

        # determine years_experience (influences growth)
        try:
            years_exp = float(user.get("years_experience", 3.0) or 3.0)
        except:
            years_exp = 3.0

        # compute an annual base growth rate influenced by experience + archetype
        annual_growth = compute_base_growth_rate(years_exp)

        # small adjust for high performers / risk_tolerance column if present
        rt = str(user.get("risk_tolerance", "")).strip().lower()
        if rt in ["high", "3", "4", "5"]:
            # risk tolerant often in gig economy -> potentially higher raises (but riskier)
            annual_growth += 0.005

        # monthly growth multipliers
        monthly_multipliers = month_growth_multiplier(annual_growth)

        # volatility tuning by income type
        if income_type == "salary":
            volatility_params = {"sigma": 0.03}
        elif income_type == "commission":
            volatility_params = {"sigma": 0.10}
        elif income_type == "freelance":
            volatility_params = {"sigma": 0.18}
        else:  # stipend
            volatility_params = {"sigma": 0.02}

        # payment day: salaries often 1-7, commissions variable, freelance random
        if income_type == "salary":
            payment_day = int(user.get("salary_day", random.randint(1, 7)) or random.randint(1, 7))
        elif income_type == "commission":
            payment_day = random.randint(1, 15)
        elif income_type == "freelance":
            payment_day = random.randint(1, 28)
        else:
            payment_day = random.randint(1, 28)

        monthly_vals = []
        months_failed = 0
        for m_idx, month in enumerate(MONTHS):
            mult = monthly_multipliers[m_idx]
            income_amount = simulate_month_income(base_income, income_type, mult, volatility_params)
            # simulate rare missed payment for salary (e.g., company delay) or freelancer (no gigs)
            if income_type == "salary":
                if random.random() < 0.01:  # 1% chance salary delayed/reduced
                    income_amount *= random.uniform(0.5, 0.95)
                    months_failed += 1
            elif income_type == "freelance":
                if random.random() < 0.08:  # chance of near-zero month
                    if random.random() < 0.5:
                        income_amount = round(income_amount * random.uniform(0.0, 0.5), 2)
                        if income_amount < 1.0:
                            months_failed += 1

            monthly_vals.append(income_amount)
            output_rows.append({
                "user_id": user_id,
                "month": month,
                "base_income": round(base_income, 2),
                "income_actual": income_amount,
                "income_type": income_type,
                "payment_day": payment_day,
                "annual_growth_rate": round(annual_growth, 4)
            })

        # compute stability metrics (per user across 12 months)
        vol_score = compute_volatility_score(monthly_vals)
        stability_index = compute_income_stability_index(vol_score, months_failed)

        # attach these metrics to the user's rows (post-hoc)
        # We'll add a small pass to update the last 12 appended rows for this user
        for i in range(12):
            output_rows[-1 - i]["volatility_score"] = vol_score
            output_rows[-1 - i]["income_stability_index"] = stability_index
            output_rows[-1 - i]["months_missed_payments"] = months_failed

    # convert to DataFrame and save
    df_out = pd.DataFrame(output_rows)
    # ensure column order
    cols_order = [
        "user_id", "month", "income_type", "payment_day", "base_income", "income_actual",
        "annual_growth_rate", "volatility_score", "income_stability_index", "months_missed_payments"
    ]
    # add any extra columns if present
    for c in cols_order:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[cols_order]

    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Income dataset generated: {OUTPUT_FILE}")
    print("Rows:", len(df_out))
    # quick summary
    print("\nSample summary (first 5 rows):")
    print(df_out.head())

if __name__ == "__main__":
    generate_income_dataset()
