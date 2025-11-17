import pandas as pd
import numpy as np
import random
from datetime import datetime

# ✅ Correct demographics dataset filename
USER_PROFILE_FILE = "users_profile_full_v3.csv"
OUTPUT_FILE = "credit_loans_12m.csv"

# Date range for 12 months
START_YEAR = 2024
START_MONTH = 5
MONTHS = 12

def generate_month_dates():
    dates = []
    for i in range(MONTHS):
        month = ((START_MONTH - 1 + i) % 12) + 1
        year = START_YEAR + ((START_MONTH - 1 + i) // 12)
        dates.append(f"{year}-{month:02d}")
    return dates

def credit_limit_from_income(income):
    return round(random.uniform(2.5, 4.5) * income, 2)

def generate_loan_amount(income, risk_tolerance, age):
    if risk_tolerance == "Low" or age < 22:
        return random.choice([0, 0, 0, round(income * random.uniform(3, 6), 2)])
    elif risk_tolerance == "Medium":
        return round(income * random.uniform(4, 10), 2)
    else:  # High risk users borrow more
        return round(income * random.uniform(8, 15), 2)

def create_credit_loan_dataset():
    df_users = pd.read_csv(USER_PROFILE_FILE)

    months = generate_month_dates()
    rows = []

    print("\n==========================================================")
    print(" Generating Credit & Loan Dataset (12 Months per User)")
    print("==========================================================\n")

    for idx, row in df_users.iterrows():
        user_id = row["user_id"]
        monthly_income = row["monthly_income"]
        risk = row["risk_tolerance"]
        age = row["age"]
        is_student = row["is_student"]
        is_metro = row["is_metro"]

        # Base probabilities influenced by demographics
        has_credit_card = (
            random.random() < 0.90 if is_metro else random.random() < 0.70
        )
        has_loan = (
            random.random() < 0.50 if age > 24 else random.random() < 0.20
        )

        credit_limit = credit_limit_from_income(monthly_income) if has_credit_card else 0
        student_credit_boost = 1.1 if is_student else 1.0
        credit_limit = round(credit_limit * student_credit_boost, 2)

        for month in months:
            outstanding_credit = (
                round(random.uniform(0.05, 0.85) * credit_limit, 2)
                if has_credit_card else 0
            )
            loan_amount = generate_loan_amount(monthly_income, risk, age) if has_loan else 0
            loan_balance = (
                round(max(loan_amount - random.uniform(0.02, 0.08) * loan_amount, 0), 2)
                if has_loan else 0
            )

            rows.append({
                "user_id": user_id,
                "month": month,
                "has_credit_card": int(has_credit_card),
                "credit_limit": credit_limit,
                "outstanding_credit": outstanding_credit,
                "credit_utilization": round(
                    outstanding_credit / credit_limit, 2
                ) if has_credit_card and credit_limit > 0 else 0,
                "has_loan": int(has_loan),
                "loan_amount": loan_amount,
                "loan_balance": loan_balance,
                "loan_to_income_ratio": round(
                    loan_balance / monthly_income, 2
                ) if loan_balance > 0 else 0
            })

        if (idx + 1) % 500 == 0:
            print(f"Processed users: {idx+1}/{len(df_users)}")

    df_output = pd.DataFrame(rows)
    df_output.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Generation Complete!")
    print(f"Total users: {len(df_users)}")
    print(f"Total rows created: {len(rows)}")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_credit_loan_dataset()
