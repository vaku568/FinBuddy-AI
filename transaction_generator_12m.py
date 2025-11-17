"""
transaction_generator_12m.py
Generates daily / transaction-level data for 12 months (May-2024 -> Apr-2025)
Input:  monthly_expenses_12m.csv
Output: transaction_data_12months.csv

Follows the reference TRANSACTION_MODEL and logic (flexible / strict modes,
dirichlet splitting, date bias patterns, merchant/payment selection).
"""

import csv
import random
import math
from datetime import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.stats import dirichlet

# -------------------------
# Configuration
# -------------------------
INPUT_FILE = "monthly_expenses_12m.csv"
OUTPUT_FILE = "transaction_data_12months.csv"
SEED = 42
FLUSH_BATCH_USERS = 500    # flush transactions to disk every N input rows (users x months processed)
LOG_EVERY = 500            # log progress every N input rows
WRITE_HEADER = True

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

# -------------------------
# TRANSACTION MODEL (kept consistent with your reference)
# -------------------------
TRANSACTION_MODEL: Dict[str, Dict] = {
    "food_expense": {
        "trans_range": (15, 50),
        "min_transaction_amount": 100,
        "allow_below_min_threshold": 150,
        "date_bias": "uniform",
        "merchants": ["Zomato", "Swiggy", "Local Restaurant", "Cafe Coffee Day", "Dominos",
                      "McDonald's", "Subway", "Haldiram's", "Street Food", "Biryani House",
                      "KFC", "Pizza Hut", "Burger King", "Food Court", "Dhaba"],
        "payment_weights": {"UPI": 0.50, "Credit Card": 0.20, "Debit Card": 0.15, "Cash": 0.15},
        "online_probability": 0.70,
        "dirichlet_alpha": 1.0
    },
    "groceries_expense": {
        "trans_range": (2, 8),
        "min_transaction_amount": 100,
        "allow_below_min_threshold": 200,
        "date_bias": "weekend",
        "specific_days": [6, 13, 20, 27],
        "merchants": ["DMart", "Big Bazaar", "Reliance Fresh", "More Supermarket",
                      "Spencer's", "Local Kirana", "24Seven", "Nature's Basket"],
        "payment_weights": {"UPI": 0.40, "Credit Card": 0.25, "Debit Card": 0.25, "Cash": 0.10},
        "online_probability": 0.30,
        "dirichlet_alpha": 2.0
    },
    "education_expense": {
        "trans_range": (1, 3),
        "min_transaction_amount": 300,
        "allow_below_min_threshold": 400,
        "date_bias": "early_month",
        "specific_days": [1, 5, 10],
        "merchants": ["Course Fee", "Udemy", "Coursera", "Books", "Study Material",
                      "Tuition", "Library", "Online Class", "Exam Fee"],
        "payment_weights": {"UPI": 0.30, "Credit Card": 0.30, "Debit Card": 0.25, "Net Banking": 0.15},
        "online_probability": 0.80,
        "dirichlet_alpha": 3.0
    },
    "subscriptions_expense": {
        "trans_range": (1, 3),
        "min_transaction_amount": 100,
        "allow_below_min_threshold": 150,
        "date_bias": "fixed",
        "specific_days": [1, 5, 15, 25],
        "merchants": ["Netflix", "Amazon Prime", "Spotify", "Hotstar", "Gym Membership",
                      "Newspaper", "Magazine", "Cloud Storage", "Software License"],
        "payment_weights": {"Credit Card": 0.50, "UPI": 0.30, "Debit Card": 0.20},
        "online_probability": 0.95,
        "dirichlet_alpha": 5.0
    },
    "fuel_expense": {
        "trans_range": (4, 10),
        "min_transaction_amount": 50,
        "allow_below_min_threshold": 100,
        "date_bias": "uniform",
        "merchants": ["HP Petrol Pump", "Indian Oil", "Bharat Petroleum", "Shell",
                      "Essar Petrol Pump", "Reliance Petrol"],
        "payment_weights": {"UPI": 0.45, "Credit Card": 0.30, "Debit Card": 0.15, "Cash": 0.10},
        "online_probability": 0.20,
        "dirichlet_alpha": 1.5
    },
    "transportation_expense": {
        "trans_range": (15, 30),
        "min_transaction_amount": 50,
        "allow_below_min_threshold": 100,
        "date_bias": "weekday",
        "merchants": ["Uber", "Ola", "Metro Card", "Auto Rickshaw", "Bus Pass",
                      "Rapido", "Local Train", "Parking", "Toll"],
        "payment_weights": {"UPI": 0.60, "Cash": 0.25, "Debit Card": 0.10, "Credit Card": 0.05},
        "online_probability": 0.75,
        "dirichlet_alpha": 0.8
    },
    "utilities_expense": {
        "trans_range": (2, 5),
        "min_transaction_amount": 100,
        "allow_below_min_threshold": 150,
        "date_bias": "early_month",
        "specific_days": [1, 3, 5, 7],
        "merchants": ["Electricity Bill", "Water Bill", "Gas Cylinder", "Internet Bill",
                      "Mobile Recharge", "DTH Recharge", "Maintenance Charge"],
        "payment_weights": {"Net Banking": 0.40, "UPI": 0.35, "Debit Card": 0.15, "Credit Card": 0.10},
        "online_probability": 0.90,
        "dirichlet_alpha": 3.5
    },
    "entertainment_expense": {
        "trans_range": (3, 10),
        "min_transaction_amount": 150,
        "allow_below_min_threshold": 250,
        "date_bias": "weekend",
        "merchants": ["BookMyShow", "PVR Cinemas", "Inox", "Gaming", "Concert Ticket",
                      "Sports Event", "Museum", "Theme Park", "Club Entry"],
        "payment_weights": {"UPI": 0.45, "Credit Card": 0.30, "Debit Card": 0.15, "Cash": 0.10},
        "online_probability": 0.85,
        "dirichlet_alpha": 1.5
    },
    "shopping_expense": {
        "trans_range": (3, 8),
        "min_transaction_amount": 100,
        "allow_below_min_threshold": 200,
        "date_bias": "mid_to_late_month",
        "specific_days": [10, 12, 15, 18, 20, 22, 25],
        "merchants": ["Amazon", "Flipkart", "Myntra", "Ajio", "Lifestyle", "Pantaloons",
                      "Westside", "Local Shop", "Decathlon", "Electronics Store"],
        "payment_weights": {"Credit Card": 0.35, "UPI": 0.40, "Debit Card": 0.15, "Cash": 0.10},
        "online_probability": 0.70,
        "dirichlet_alpha": 1.2
    },
    "healthcare_expense": {
        "trans_range": (1, 3),
        "min_transaction_amount": 350,
        "allow_below_min_threshold": 500,
        "date_bias": "random",
        "merchants": ["Apollo Pharmacy", "MedPlus", "Doctor Consultation", "Lab Test",
                      "Hospital", "Dental Clinic", "Physiotherapy", "Health Checkup"],
        "payment_weights": {"UPI": 0.35, "Cash": 0.30, "Credit Card": 0.20, "Debit Card": 0.15},
        "online_probability": 0.40,
        "dirichlet_alpha": 2.5
    },
    "personal_care_expense": {
        "trans_range": (2, 6),
        "min_transaction_amount": 270,
        "allow_below_min_threshold": 400,
        "date_bias": "weekend",
        "merchants": ["Salon", "Spa", "Barber Shop", "Beauty Parlor", "Cosmetics",
                      "Personal Care Products", "Grooming", "Wellness Center"],
        "payment_weights": {"UPI": 0.40, "Cash": 0.35, "Credit Card": 0.15, "Debit Card": 0.10},
        "online_probability": 0.35,
        "dirichlet_alpha": 2.0
    },
    "miscellaneous_expense": {
        "trans_range": (5, 20),
        "min_transaction_amount": 150,
        "allow_below_min_threshold": 250,
        "date_bias": "uniform",
        "merchants": ["Gifts", "Charity", "Emergency", "Repairs", "Services",
                      "Miscellaneous", "Courier", "ATM Withdrawal", "Bank Charges"],
        "payment_weights": {"UPI": 0.40, "Cash": 0.30, "Credit Card": 0.15, "Debit Card": 0.15},
        "online_probability": 0.50,
        "dirichlet_alpha": 0.7
    }
}

EXPENSE_CATEGORIES = list(TRANSACTION_MODEL.keys())

# -------------------------
# Helper functions
# -------------------------
def generate_smart_transaction_amounts(total_amount: float, num_trans: int,
                                       alpha: float, min_amount: float,
                                       threshold: float) -> List[int]:
    """
    Produce integer transaction amounts summing to total_amount.
    Implements flexible (small totals) and strict modes per reference.
    """
    total_amount = int(round(total_amount))
    if total_amount <= 0:
        return []

    if num_trans <= 1:
        return [total_amount]

    # Flexible mode
    if total_amount < threshold:
        # allow smaller than min_amount — split via Dirichlet
        props = dirichlet.rvs([alpha] * num_trans, size=1)[0]
        amts = [int(round(p * total_amount)) for p in props]
        diff = total_amount - sum(amts)
        if diff != 0:
            amts[amts.index(max(amts))] += diff
        # remove zeros (if any) by merging into biggest
        if any(a == 0 for a in amts):
            # ensure at least 1 transaction gets something
            for i, a in enumerate(amts):
                if a == 0:
                    amts[amts.index(max(amts))] += 0  # no-op but keep structure
        return amts

    # Strict mode: reserve min_amount per transaction
    min_total_needed = num_trans * min_amount
    if total_amount < min_total_needed:
        # reduce num_trans to feasible number
        feasible = max(1, total_amount // min_amount)
        if feasible == 0:
            return [total_amount]
        num_trans = feasible

    reserved = num_trans * min_amount
    remaining = total_amount - reserved
    if remaining <= 0:
        base = [min_amount] * num_trans
        diff = total_amount - sum(base)
        if diff != 0:
            base[-1] += diff
        return base

    props = dirichlet.rvs([alpha] * num_trans, size=1)[0]
    extras = [int(round(p * remaining)) for p in props]
    amounts = [min_amount + e for e in extras]
    diff = total_amount - sum(amounts)
    if diff != 0:
        amounts[amounts.index(max(amounts))] += diff
    return amounts

def get_days_in_month(year: int, month: int) -> int:
    if month == 12:
        return (datetime(year + 1, 1, 1) - datetime(year, month, 1)).days
    return (datetime(year, month + 1, 1) - datetime(year, month, 1)).days

def generate_dates_with_bias(year: int, month: int, num_trans: int,
                             date_bias: str, config: dict) -> List[datetime]:
    days_in_month = get_days_in_month(year, month)
    dates = []
    if num_trans <= 0:
        return dates

    if date_bias == "fixed" and "specific_days" in config:
        available_days = [d for d in config["specific_days"] if d <= days_in_month]
        # repeat available days if needed
        chosen = []
        while len(chosen) < num_trans:
            chosen.extend(random.sample(available_days, k=min(len(available_days), num_trans - len(chosen))))
            if not available_days:
                break
        for d in chosen[:num_trans]:
            dates.append(datetime(year, month, d, random.randint(0, 23), random.randint(0, 59)))

    elif date_bias == "early_month":
        for _ in range(num_trans):
            day = random.randint(1, min(10, days_in_month))
            dates.append(datetime(year, month, day, random.randint(8, 20), random.randint(0, 59)))

    elif date_bias == "weekend":
        # try to bias to weekends; fallback to random if cannot find enough
        tries = 0
        while len(dates) < num_trans and tries < num_trans * 5:
            d = random.randint(1, days_in_month)
            dt = datetime(year, month, d, random.randint(10, 21), random.randint(0, 59))
            if dt.weekday() >= 5 or random.random() < 0.3:  # some probability to accept non-weekend
                dates.append(dt)
            tries += 1
        while len(dates) < num_trans:
            d = random.randint(1, days_in_month)
            dates.append(datetime(year, month, d, random.randint(10, 21), random.randint(0, 59)))

    elif date_bias == "weekday":
        tries = 0
        while len(dates) < num_trans and tries < num_trans * 5:
            d = random.randint(1, days_in_month)
            dt = datetime(year, month, d, random.randint(7, 20), random.randint(0, 59))
            if dt.weekday() < 5 or random.random() < 0.2:
                dates.append(dt)
            tries += 1
        while len(dates) < num_trans:
            d = random.randint(1, days_in_month)
            dates.append(datetime(year, month, d, random.randint(7, 20), random.randint(0, 59)))

    elif date_bias == "mid_to_late_month":
        for _ in range(num_trans):
            day = random.randint(10, min(25, days_in_month))
            dates.append(datetime(year, month, day, random.randint(11, 21), random.randint(0, 59)))

    else:  # uniform / random
        for _ in range(num_trans):
            day = random.randint(1, days_in_month)
            dates.append(datetime(year, month, day, random.randint(6, 23), random.randint(0, 59)))

    dates = sorted(dates)
    return dates

def select_payment_method(weights: dict) -> str:
    methods = list(weights.keys())
    probs = list(weights.values())
    return random.choices(methods, weights=probs, k=1)[0]

def generate_transactions_for_category(user_id: str, month: int, year: int,
                                       category: str, total_amount: float,
                                       user_profile: dict) -> List[Dict]:
    """
    Returns list of transactions (dicts with date, amount, merchant, payment_method, is_online, category)
    """
    if total_amount <= 0 or category not in TRANSACTION_MODEL:
        return []

    cfg = TRANSACTION_MODEL[category]
    min_amt = cfg["min_transaction_amount"]
    threshold = cfg["allow_below_min_threshold"]
    min_trans, max_trans = cfg["trans_range"]

    # determine num_trans based on total & archetype
    if total_amount < threshold:
        num_trans = random.randint(1, max(1, min_trans // 2))
    else:
        max_possible = max(1, int(total_amount // min_amt))
        max_t = min(max_trans, max_possible)
        min_t = min(min_trans, max_possible)
        archetype = user_profile.get("user_archetype", "")
        if archetype == "impulsive_spender":
            num_trans = random.randint(min_t, min_t + max(0, (max_t - min_t) // 2))
        elif archetype in ["meticulous_tracker", "balanced_planner"]:
            num_trans = random.randint(min_t + max(0, (max_t - min_t) // 2), max_t)
        else:
            num_trans = random.randint(min_t, max_t)

    num_trans = max(1, num_trans)

    # get integer amounts
    alpha = cfg.get("dirichlet_alpha", 1.0)
    amounts = generate_smart_transaction_amounts(total_amount, num_trans, alpha, min_amt, threshold)

    # generate dates with bias (use year/month)
    dates = generate_dates_with_bias(year, month, len(amounts), cfg.get("date_bias", "uniform"), cfg)

    transactions = []
    for i, amt in enumerate(amounts):
        merchant = random.choice(cfg["merchants"])
        payment_method = select_payment_method(cfg["payment_weights"])
        is_online = random.random() < cfg["online_probability"]
        date_obj = dates[i] if i < len(dates) else dates[-1] if dates else datetime(year, month, 1, 12, 0)
        transactions.append({
            "date": date_obj,
            "amount": int(amt),
            "merchant": merchant,
            "payment_method": payment_method,
            "is_online": bool(is_online),
            "category": category.replace("_expense", "")
        })
    return transactions

# -------------------------
# Main streaming generation
# -------------------------
def generate_all_transactions_stream(input_csv: str = INPUT_FILE, output_csv: str = OUTPUT_FILE):
    # Read monthly file
    df = pd.read_csv(input_csv, dtype={"user_id": object})
    required_cols = {"user_id", "month_index", "month_start_date"}.union(set(EXPENSE_CATEGORIES))
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in monthly CSV: {missing}")

    # open writer
    fieldnames = ["transaction_id", "user_id", "date", "month_index", "category",
                  "merchant", "amount", "payment_method", "is_online", "description"]
    # write header once
    with open(output_csv, "w", newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

    transaction_counter = 1
    stats = {"flexible": 0, "strict": 0, "tx_count": 0, "rows_processed": 0}
    buffer_rows = []

    # iterate rows (each row is user-month)
    for idx, row in df.iterrows():
        # month_start_date should exist (Option B mapping)
        try:
            month_start = datetime.strptime(str(row["month_start_date"]), "%Y-%m-%d")
        except Exception:
            # fallback: infer year/month from month_index mapping: month_index 1 -> May 2024 etc
            # map month_index (1..12) -> months May-2024 .. Apr-2025
            m_idx = int(row["month_index"])
            # mapping: 1->May2024, 2->Jun2024, ... 12->Apr2025
            month_order = [ (2024,5), (2024,6), (2024,7), (2024,8), (2024,9), (2024,10),
                            (2024,11),(2024,12),(2025,1),(2025,2),(2025,3),(2025,4) ]
            if 1 <= m_idx <= 12:
                year, monthnum = month_order[m_idx - 1]
                month_start = datetime(year, monthnum, 1)
            else:
                raise ValueError("month_index out of range and month_start_date parse failed")

        year = month_start.year
        monthnum = month_start.month

        user_id = str(row["user_id"])
        user_profile = {
            "user_archetype": row.get("user_archetype", ""),
            "risk_tolerance": row.get("risk_tolerance", None),
            "is_metro": row.get("is_metro", None)
        }

        # iterate expense categories and create transactions
        for cat in EXPENSE_CATEGORIES:
            total_amount = float(row.get(cat, 0) if not pd.isna(row.get(cat, 0)) else 0)
            if total_amount <= 0:
                continue

            # compute transactions for this category
            trans_list = generate_transactions_for_category(user_id, monthnum, year, cat, total_amount, user_profile)

            # classification flexible vs strict (for stats)
            threshold = TRANSACTION_MODEL[cat]["allow_below_min_threshold"]
            if total_amount < threshold:
                stats["flexible"] += 1
            else:
                stats["strict"] += 1

            for t in trans_list:
                out_row = {
                    "transaction_id": f"TXN-{transaction_counter:08d}",
                    "user_id": user_id,
                    "date": t["date"].strftime("%Y-%m-%d %H:%M:%S"),
                    "month_index": int(row["month_index"]),
                    "category": t["category"],
                    "merchant": t["merchant"],
                    "amount": int(t["amount"]),
                    "payment_method": t["payment_method"],
                    "is_online": int(t["is_online"]),
                    "description": f"{t['merchant']} - {t['category']}"
                }
                buffer_rows.append(out_row)
                transaction_counter += 1
                stats["tx_count"] += 1

        stats["rows_processed"] += 1

        # flush buffer to CSV periodically to avoid large memory use
        if stats["rows_processed"] % FLUSH_BATCH_USERS == 0 or idx == len(df) - 1:
            with open(output_csv, "a", newline='', encoding='utf-8') as fout:
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                writer.writerows(buffer_rows)
            buffer_rows = []

        # progress logging
        if stats["rows_processed"] % LOG_EVERY == 0:
            print(f"Processed rows (user-months): {stats['rows_processed']}/{len(df)}  | Transactions so far: {stats['tx_count']:,}")

    # final flush (if any)
    if buffer_rows:
        with open(output_csv, "a", newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writerows(buffer_rows)
        buffer_rows = []

    # Print summary
    print("\n=== GENERATION COMPLETE ===")
    print(f"User-month rows processed : {stats['rows_processed']}")
    print(f"Total transactions generated: {stats['tx_count']:,}")
    print(f"Flexible-mode categories: {stats['flexible']}")
    print(f"Strict-mode categories: {stats['strict']}")
    print(f"Output CSV: {output_csv}")
    return

# -------------------------
# Run as script
# -------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Transaction generator (12 months) - Option B mapping (May 2024 -> Apr 2025)")
    print("=" * 60)
    generate_all_transactions_stream(INPUT_FILE, OUTPUT_FILE)
