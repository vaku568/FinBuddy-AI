"""
financial_goals_12m.py

Generates realistic financial goals per user (1-4 goals each) using both
Standard Core Goals (A) and Expanded Goals (B).

Output:
  - financial_goals_12months.csv

Uses user profile file (auto-detect):
  - users_profile_full_v3.csv
  - users_profile_full.csv
  - user_profile_10k.csv
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

POSSIBLE_USER_FILES = [
    "users_profile_full_v3.csv",
    "users_profile_full.csv",
    "user_profile_10k.csv",
    "users_profile_full_v3.xlsx",
    "users_profile_full.xlsx"
]

OUTPUT_FILE = "financial_goals_12months.csv"

# Combined goal types (A + B)
GOAL_TYPES = [
    "Emergency Fund", "Travel", "Education", "Laptop/Mobile Upgrade",
    "Vehicle (Bike)", "Vehicle (Car)", "Home Down Payment",
    "Marriage", "Retirement", "Business Startup", "Health/Insurance Fund",
    "Luxury Purchase", "Investment Wealth Goal"
]

# Priority defaults by goal (1 low - 10 high)
DEFAULT_PRIORITY = {
    "Emergency Fund": 10,
    "Travel": 5,
    "Education": 8,
    "Laptop/Mobile Upgrade": 4,
    "Vehicle (Bike)": 5,
    "Vehicle (Car)": 7,
    "Home Down Payment": 9,
    "Marriage": 8,
    "Retirement": 10,
    "Business Startup": 7,
    "Health/Insurance Fund": 9,
    "Luxury Purchase": 3,
    "Investment Wealth Goal": 8
}

RISK_BUCKETS = ["Low", "Medium", "High"]

# Helpful helpers
def find_user_file():
    for f in POSSIBLE_USER_FILES:
        if os.path.exists(f):
            return f
    raise FileNotFoundError("No user profile file found. Looked for: " + ", ".join(POSSIBLE_USER_FILES))

def safe_get(row, key, default=None):
    return row.get(key) if key in row else default

def years_to_target_by_goal(goal_type, age, archetype):
    # Rough default horizon in years
    if goal_type == "Emergency Fund":
        return random.choice([1, 1, 2])  # short-term
    if goal_type == "Travel":
        return random.choice([0.5, 1, 2])
    if goal_type == "Education":
        return random.choice([1, 2, 3])
    if goal_type == "Laptop/Mobile Upgrade":
        return random.choice([0.25, 0.5, 1])
    if goal_type.startswith("Vehicle"):
        return random.choice([1, 2, 3])
    if goal_type == "Home Down Payment":
        return random.choice([3, 5, 7])
    if goal_type == "Marriage":
        return random.choice([1, 2, 3])
    if goal_type == "Retirement":
        # use age to determine horizon
        retire_age = 60
        yrs = max(5, (retire_age - (age or 30)))
        return max(5, int(min(40, yrs)))
    if goal_type == "Business Startup":
        return random.choice([1, 2, 3, 4])
    if goal_type == "Health/Insurance Fund":
        return random.choice([1, 2])
    if goal_type == "Luxury Purchase":
        return random.choice([0.5, 1, 2])
    if goal_type == "Investment Wealth Goal":
        return random.choice([3, 5, 7, 10])
    return 2

def estimate_target_amount(goal_type, monthly_income, monthly_expenses, age):
    # Base heuristics using income and expenses
    income = max(1.0, float(monthly_income or 0.0))
    expenses = max(0.0, float(monthly_expenses or income * 0.6))
    if goal_type == "Emergency Fund":
        # aim 3-6 months of expenses
        months = random.choice([3, 4, 6])
        return round(expenses * months, 2)
    if goal_type == "Travel":
        return round(income * random.uniform(2, 6), 2)
    if goal_type == "Education":
        # if younger -> smaller (course), if older -> higher (higher studies)
        factor = random.uniform(6, 24) if age and age < 30 else random.uniform(12, 36)
        return round(income * factor, 2)
    if goal_type == "Laptop/Mobile Upgrade":
        return round(random.choice([40000, 70000, 120000]) * (1 if income>0 else 1), 2)
    if goal_type == "Vehicle (Bike)":
        return round(random.uniform(40000, 150000), 2)
    if goal_type == "Vehicle (Car)":
        return round(random.uniform(400000, 2000000), 2)
    if goal_type == "Home Down Payment":
        # assume 20% down of an avg house price (varies by income)
        avg_house_price = max(2000000, income * 50 * 12)  # rough proxy
        return round(avg_house_price * random.uniform(0.15, 0.30), 2)
    if goal_type == "Marriage":
        return round(income * random.uniform(24, 120), 2)
    if goal_type == "Retirement":
        # want replacement of annual income * 20-30
        annual = income * 12
        return round(annual * random.uniform(15, 30), 2)
    if goal_type == "Business Startup":
        return round(income * random.uniform(24, 120), 2)
    if goal_type == "Health/Insurance Fund":
        return round(income * random.uniform(6, 24), 2)
    if goal_type == "Luxury Purchase":
        return round(income * random.uniform(6, 36), 2)
    if goal_type == "Investment Wealth Goal":
        return round(income * random.uniform(60, 300), 2)
    return round(income * 12, 2)

def choose_risk_by_goal(goal_type, archetype):
    # Simple mapping: emergency very low risk, business high risk, investment wealth high
    if goal_type in ["Emergency Fund", "Health/Insurance Fund"]:
        return "Low"
    if goal_type in ["Business Startup", "Investment Wealth Goal"]:
        return "High"
    if archetype and "impulsive" in archetype.lower():
        return random.choice(["Medium", "High"])
    return random.choice(["Low", "Medium"])

def months_to_target(years):
    return max(1, int(round(years * 12)))

def generate_goal_id(seq):
    return f"G-{seq:08d}"

def clamp_priority(p):
    return max(1, min(10, int(round(p))))

def pick_goal_types_for_user(age, archetype, is_student, income, existing_goals_count):
    # pick 1-4 goals, prefer some based on demographics
    n = random.choices([1,2,3,4], weights=[0.2,0.45,0.25,0.10])[0]
    # ensure not too many for low-income users
    if income is None or income <= 0:
        n = min(n, 2)
    # priority picks
    candidates = GOAL_TYPES.copy()
    picks = set()

    # always include emergency fund with some probability
    if random.random() < 0.85:
        picks.add("Emergency Fund")

    # students: education, laptop/mobile, travel
    if is_student:
        picks.add("Education")
        if random.random() < 0.6:
            picks.add("Laptop/Mobile Upgrade")
        if random.random() < 0.4:
            picks.add("Travel")

    if age and age > 30:
        # increased chance of home / retirement / vehicle / marriage
        if random.random() < 0.4:
            picks.add("Home Down Payment")
        if random.random() < 0.6:
            picks.add("Retirement")

    # archetype biases
    if archetype:
        a = archetype.lower()
        if "impulsive" in a:
            picks.update(["Travel", "Luxury Purchase"])
        if "goal" in a or "planner" in a or "optimizer" in a:
            picks.update(["Home Down Payment", "Investment Wealth Goal", "Retirement"])
        if "aggressive" in a or "investor" in a:
            picks.update(["Investment Wealth Goal", "Business Startup"])

    # fill random choices until n reached
    while len(picks) < n:
        picks.add(random.choice(candidates))

    # convert to list and trim to n
    picks = list(picks)[:n]
    return picks

def generate_financial_goals():
    user_file = find_user_file()
    print("Loading users from:", user_file)
    if user_file.endswith(".xlsx"):
        df_users = pd.read_excel(user_file)
    else:
        df_users = pd.read_csv(user_file)

    # Validate minimal columns
    if "user_id" not in df_users.columns:
        raise KeyError("User profile file must contain 'user_id' column")
    # optional columns: monthly_income, monthly_expenses, age, user_archetype, savings_rate, is_student
    rows = []
    seq = 1
    now = datetime.utcnow().strftime("%Y-%m-%d")

    for idx, u in df_users.iterrows():
        urow = u.to_dict()
        user_id = urow.get("user_id")
        age = int(safe_get(urow, "age", 30) or 30)
        archetype = safe_get(urow, "user_archetype", "") or ""
        is_student = str(safe_get(urow, "is_student", "No")).strip().lower() in ["yes","true","1"]
        monthly_income = float(safe_get(urow, "monthly_income", 0.0) or 0.0)
        monthly_expenses = float(safe_get(urow, "monthly_expenses", max(0.0, monthly_income*0.6)) or 0.0)
        savings_rate = float(safe_get(urow, "savings_rate", 0.0) or 0.0)

        # choose 1-4 goals
        goal_types = pick_goal_types_for_user(age, archetype, is_student, monthly_income, 0)

        for g in goal_types:
            years = years_to_target_by_goal(g, age, archetype)
            months = months_to_target(years)
            target_amount = estimate_target_amount(g, monthly_income, monthly_expenses, age)
            # ensure target > 0
            target_amount = max(1000.0, float(target_amount))

            # current saved (some fraction of target; more for planners)
            base_saved_frac = 0.10
            if "planner" in archetype.lower() or "goal" in archetype.lower() or "optimizer" in archetype.lower():
                base_saved_frac = random.uniform(0.15, 0.40)
            if is_student:
                base_saved_frac = random.uniform(0.01, 0.10)
            current_saved = round(target_amount * base_saved_frac * random.uniform(0.5, 1.0), 2)

            # monthly commitment: try to use user's savings capacity first
            # savings capacity estimated as monthly_income * savings_rate (if present)
            savings_capacity = monthly_income * (savings_rate if savings_rate>0 else 0.15)
            # required monthly = (target - current_saved) / months
            required_monthly = max(0.0, (target_amount - current_saved) / months)
            # realistic commitment: min(required_monthly, savings_capacity * factor)
            factor = random.uniform(0.5, 1.2)
            monthly_commitment = round(min(required_monthly, max(0.0, savings_capacity * factor)), 2)

            # if commitment is zero (no capacity), set a very small commitment as placeholder
            if monthly_commitment <= 0:
                monthly_commitment = round(max(0.0, required_monthly * random.uniform(0.05, 0.15)), 2)

            # target date: today + months
            target_date = (datetime.utcnow() + timedelta(days=months*30)).strftime("%Y-%m-%d")

            # priority: base + modifiers
            priority = DEFAULT_PRIORITY.get(g, 5)
            # nudge priority if current_saved low relative to target, or if emergency
            if g == "Emergency Fund":
                if current_saved < target_amount * 0.5:
                    priority += 1
            # archetype adjustments
            if "impulsive" in archetype.lower() and g == "Travel":
                priority += 1
            priority = clamp_priority(priority + random.randint(-2,2))

            # risk bucket for goal
            risk_bucket = choose_risk_by_goal(g, archetype)

            # progress percent
            progress_percent = round(100.0 * current_saved / target_amount, 2)

            # textual description
            description = f"{g} target of ₹{target_amount:,.0f}"

            rows.append({
                "goal_id": generate_goal_id(seq),
                "user_id": user_id,
                "goal_type": g,
                "goal_description": description,
                "target_amount": round(target_amount,2),
                "current_saved": round(current_saved,2),
                "monthly_commitment": monthly_commitment,
                "months_to_target": months,
                "target_date": target_date,
                "goal_created_date": now,
                "priority_score": priority,
                "risk_category": risk_bucket,
                "progress_percent": progress_percent
            })
            seq += 1

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_FILE, index=False)
    # summary
    print("\n✅ Financial Goals dataset generated:", OUTPUT_FILE)
    print("Total users:", len(df_users))
    print("Total goals rows:", len(df_out))
    print("\nSample (first 8 rows):")
    print(df_out.head(8).to_string(index=False))
    # simple stats
    by_goal = df_out['goal_type'].value_counts().head(12)
    print("\nGoal type distribution (top):")
    print(by_goal)

if __name__ == "__main__":
    generate_financial_goals()
