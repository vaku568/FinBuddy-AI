"""
mysql_bulk_loader_fixed.py

Robust CSV -> MySQL loader tuned to the CSV headers you provided.
- Creates tables if not exists (matching CSV headers)
- Copies CSVs to MySQL secure_file_priv folder
- Uses LOAD DATA INFILE with explicit column lists and SET transforms
- Temporarily relaxes strict sql_mode to avoid rejects on empty strings
"""

import os
import shutil
import time
import mysql.connector
from mysql.connector import errorcode

# -----------------------
# EDIT THESE BEFORE RUN
# -----------------------
SRC_DATA_DIR = r"C:\Users\Lenovo\Downloads\dataset"
UPLOAD_DIR = r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads"
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASS = "root"   # <-- update if needed
DB_NAME = "finbuddy_db"
# -----------------------

CSV_TABLE_MAP = {
    "users_profile_full_v3.csv": "users",
    "monthly_expenses_12m.csv": "monthly_expenses",
    "transaction_data_12months.csv": "transactions",
    "credit_loans_12m.csv": "credit_loans",
    "investment_data_12months.csv": "investments",
    "income_data_12months.csv": "income",
    "financial_goals_12months.csv": "financial_goals",
    "fraud_signals_12months.csv": "fraud_signals",
    # 👇 NEW
    "subscriptions_12months.csv": "subscriptions"
}

# -----------------------
# CREATE TABLE DDL (aligned with your CSV headers)
# -----------------------
CREATE_TABLE_SQL = {
    "users": """
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(32) PRIMARY KEY,
        age INT,
        monthly_income DECIMAL(12,2),
        education VARCHAR(100),
        city VARCHAR(100),
        is_metro TINYINT,
        dependents INT,
        years_experience INT,
        is_student TINYINT,
        risk_tolerance VARCHAR(50),
        monthly_expenses DECIMAL(12,2),
        food_expense DECIMAL(12,2),
        groceries_expense DECIMAL(12,2),
        education_expense DECIMAL(12,2),
        subscriptions_expense DECIMAL(12,2),
        fuel_expense DECIMAL(12,2),
        transportation_expense DECIMAL(12,2),
        utilities_expense DECIMAL(12,2),
        entertainment_expense DECIMAL(12,2),
        shopping_expense DECIMAL(12,2),
        healthcare_expense DECIMAL(12,2),
        personal_care_expense DECIMAL(12,2),
        miscellaneous_expense DECIMAL(12,2),
        monthly_surplus DECIMAL(12,2),
        savings_rate DECIMAL(6,4),
        investment_amount DECIMAL(12,2),
        has_investments TINYINT,
        technology_comfort VARCHAR(80),
        money_management_approach VARCHAR(120),
        decision_making_style VARCHAR(120),
        goal_setting_behavior VARCHAR(80),
        preferred_communication VARCHAR(80),
        information_processing VARCHAR(80),
        user_archetype VARCHAR(80),
        debt_to_income DECIMAL(6,3)
    ) ENGINE=InnoDB;
    """,

    "monthly_expenses": """
    CREATE TABLE IF NOT EXISTS monthly_expenses (
        user_id VARCHAR(32),
        month DATE,
        month_index INT,
        month_start_date DATE,
        month_name VARCHAR(16),
        monthly_income DECIMAL(12,2),
        monthly_expenses DECIMAL(12,2),
        monthly_surplus DECIMAL(12,2),
        savings_rate DECIMAL(6,4),
        user_archetype VARCHAR(80),
        is_metro TINYINT,
        is_student TINYINT,
        age INT,
        food_expense DECIMAL(12,2),
        groceries_expense DECIMAL(12,2),
        education_expense DECIMAL(12,2),
        subscriptions_expense DECIMAL(12,2),
        fuel_expense DECIMAL(12,2),
        transportation_expense DECIMAL(12,2),
        utilities_expense DECIMAL(12,2),
        entertainment_expense DECIMAL(12,2),
        shopping_expense DECIMAL(12,2),
        healthcare_expense DECIMAL(12,2),
        personal_care_expense DECIMAL(12,2),
        miscellaneous_expense DECIMAL(12,2),
        PRIMARY KEY (user_id, month)
    ) ENGINE=InnoDB;
    """,

    "transactions": """
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(32),
        date_time DATETIME,
        month_index INT,
        category VARCHAR(80),
        merchant VARCHAR(150),
        amount DECIMAL(12,2),
        payment_method VARCHAR(80),
        is_online TINYINT,
        description VARCHAR(255),
        INDEX idx_tr_user_date (user_id, date_time),
        INDEX idx_tr_month (month_index)
    ) ENGINE=InnoDB;
    """,

    "credit_loans": """
    CREATE TABLE IF NOT EXISTS credit_loans (
        user_id VARCHAR(32),
        month DATE,
        has_credit_card TINYINT,
        credit_limit DECIMAL(12,2),
        outstanding_credit DECIMAL(12,2),
        credit_utilization DECIMAL(6,3),
        has_loan TINYINT,
        loan_amount DECIMAL(12,2),
        loan_balance DECIMAL(12,2),
        loan_to_income_ratio DECIMAL(6,3),
        PRIMARY KEY (user_id, month)
    ) ENGINE=InnoDB;
    """,

    "investments": """
    CREATE TABLE IF NOT EXISTS investments (
        user_id VARCHAR(32),
        month DATE,
        investment_monthly DECIMAL(12,2),
        stocks DECIMAL(12,2),
        sip DECIMAL(12,2),
        crypto DECIMAL(12,2),
        gold_bonds DECIMAL(12,2),
        skip_month TINYINT,
        total_investment_value DECIMAL(14,2),
        PRIMARY KEY (user_id, month)
    ) ENGINE=InnoDB;
    """,

    "income": """
    CREATE TABLE IF NOT EXISTS income (
        user_id VARCHAR(32),
        month DATE,
        income_type VARCHAR(32),
        payment_day INT,
        base_income DECIMAL(12,2),
        income_actual DECIMAL(12,2),
        annual_growth_rate DECIMAL(6,3),
        volatility_score DECIMAL(6,3),
        income_stability_index DECIMAL(6,3),
        months_missed_payments INT,
        PRIMARY KEY (user_id, month)
    ) ENGINE=InnoDB;
    """,

    "financial_goals": """
    CREATE TABLE IF NOT EXISTS financial_goals (
        goal_id VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(32),
        goal_type VARCHAR(80),
        goal_description VARCHAR(255),
        target_amount DECIMAL(14,2),
        current_saved DECIMAL(14,2),
        monthly_commitment DECIMAL(12,2),
        months_to_target INT,
        target_date DATE,
        goal_created_date DATE,
        priority_score INT,
        risk_category VARCHAR(32),
        progress_percent DECIMAL(6,3)
    ) ENGINE=InnoDB;
    """,

    "fraud_signals": """
    CREATE TABLE IF NOT EXISTS fraud_signals (
        transaction_id VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(32),
        date_time DATETIME,
        fraud_type VARCHAR(100),
        severity VARCHAR(32),
        flagged_amount DECIMAL(12,2),
        fraud_label VARCHAR(64)
    ) ENGINE=InnoDB;
    """,

    # 👇 NEW: Subscriptions
    "subscriptions": """
    CREATE TABLE IF NOT EXISTS subscriptions (
        user_id VARCHAR(32),
        month DATE,
        active_subs INT,
        canceled_subs INT,
        avg_sub_fee DECIMAL(10,2),
        total_fee_paid DECIMAL(12,2),
        auto_renew_flag TINYINT,
        churn_flag TINYINT,
        churn_rate DECIMAL(6,3),
        subs_to_fee_ratio DECIMAL(10,4),
        PRIMARY KEY (user_id, month)
    ) ENGINE=InnoDB;
    """
}

# -----------------------
# Explicit LOAD statements per CSV
# -----------------------
LOAD_SQL = {
    "users_profile_full_v3.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE users
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (user_id, age, @month_ignore, monthly_income, education, city, is_metro, dependents, years_experience,
         is_student, risk_tolerance, monthly_expenses, food_expense, groceries_expense, education_expense,
         subscriptions_expense, fuel_expense, transportation_expense, utilities_expense, entertainment_expense,
         shopping_expense, healthcare_expense, personal_care_expense, miscellaneous_expense, monthly_surplus,
         savings_rate, investment_amount, has_investments, technology_comfort, money_management_approach,
         decision_making_style, goal_setting_behavior, preferred_communication, information_processing,
         user_archetype, debt_to_income)
        SET
          is_metro = IF(TRIM(is_metro) IN ('1','yes','Yes','YES','Y'),1,0),
          is_student = IF(TRIM(is_student) IN ('1','yes','Yes','YES','Y'),1,0)
        ;
        """
    ),

    "monthly_expenses_12m.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE monthly_expenses
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (user_id, month_index, month_start_date, month_name, monthly_income, monthly_expenses,
         monthly_surplus, savings_rate, user_archetype, is_metro, is_student, age,
         food_expense, groceries_expense, education_expense, subscriptions_expense, fuel_expense,
         transportation_expense, utilities_expense, entertainment_expense, shopping_expense,
         healthcare_expense, personal_care_expense, miscellaneous_expense)
        SET month = STR_TO_DATE(month_start_date, '%Y-%m-%d'),
            is_metro = IF(TRIM(is_metro) IN ('1','yes','Yes','YES','Y'),1,0),
            is_student = IF(TRIM(is_student) IN ('1','yes','Yes','YES','Y'),1,0)
        ;
        """
    ),

    "transaction_data_12months.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE transactions
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (transaction_id, user_id, @date_str, month_index, category, merchant, amount, payment_method, is_online, description)
        SET date_time = STR_TO_DATE(@date_str, '%Y-%m-%d %H:%i:%s'),
            is_online = IF(TRIM(is_online) IN ('1','True','true','TRUE','yes','y'),1,0)
        ;
        """
    ),

    "credit_loans_12m.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE credit_loans
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (user_id, @month_str, has_credit_card, credit_limit, outstanding_credit, credit_utilization,
         has_loan, loan_amount, loan_balance, loan_to_income_ratio)
        SET month = STR_TO_DATE(CONCAT(@month_str,'-01'), '%Y-%m-%d'),
            has_credit_card = IF(TRIM(has_credit_card) IN ('1','yes','Yes','Y'),1,0),
            has_loan = IF(TRIM(has_loan) IN ('1','yes','Yes','Y'),1,0)
        ;
        """
    ),

    "investment_data_12months.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE investments
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (user_id, @month_str, investment_monthly, stocks, sip, crypto, gold_bonds, skip_month, total_investment_value)
        SET month = STR_TO_DATE(CONCAT(@month_str,'-01'), '%Y-%m-%d'),
            skip_month = IF(TRIM(skip_month) IN ('1','yes','Yes','Y'),1,0)
        ;
        """
    ),

    "income_data_12months.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE income
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (user_id, @month_str, income_type, payment_day, base_income, income_actual, annual_growth_rate,
         volatility_score, income_stability_index, months_missed_payments)
        SET month = STR_TO_DATE(CONCAT(@month_str,'-01'), '%Y-%m-%d')
        ;
        """
    ),

    "financial_goals_12months.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE financial_goals
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (goal_id, user_id, goal_type, goal_description, target_amount, current_saved, monthly_commitment,
         months_to_target, @target_dt, @goal_created_dt, priority_score, risk_category, progress_percent)
        SET target_date = STR_TO_DATE(@target_dt, '%Y-%m-%d'),
            goal_created_date = STR_TO_DATE(@goal_created_dt, '%Y-%m-%d')
        ;
        """
    ),

    "fraud_signals_12months.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE fraud_signals
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (transaction_id, user_id, @dt, fraud_type, severity, flagged_amount, fraud_label)
        SET date_time = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s')
        ;
        """
    ),

    # 👇 NEW: Subscriptions loader
    "subscriptions_12months.csv": lambda upload_path: (
        f"""
        LOAD DATA INFILE '{upload_path}'
        INTO TABLE subscriptions
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (user_id, @month_str, active_subs, canceled_subs, avg_sub_fee, total_fee_paid,
         auto_renew_flag, churn_flag, churn_rate, subs_to_fee_ratio)
        SET month = STR_TO_DATE(CONCAT(@month_str,'-01'), '%Y-%m-%d'),
            auto_renew_flag = IF(TRIM(auto_renew_flag) IN ('1','yes','Yes','Y'),1,0),
            churn_flag = IF(TRIM(churn_flag) IN ('1','yes','Yes','Y'),1,0)
        ;
        """
    ),
}

# -----------------------
# Helper functions
# -----------------------
def copy_csv_to_upload(csv_filename):
    src = os.path.join(SRC_DATA_DIR, csv_filename)
    dest = os.path.join(UPLOAD_DIR, csv_filename)
    if not os.path.exists(src):
        print(f"[WARN] Source file not found: {src}  — skipping")
        return False
    try:
        shutil.copy2(src, dest)
        print(f"[OK] Copied: {csv_filename} -> Upload folder")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to copy {csv_filename}: {e}")
        return False

def connect_db():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME,
            autocommit=True, allow_local_infile=True
        )
        return conn
    except mysql.connector.Error as err:
        print("ERROR: Could not connect to DB:", err)
        raise

def create_tables(conn):
    cur = conn.cursor()
    for name, ddl in CREATE_TABLE_SQL.items():
        print(f"Creating table `{name}` ...", end=" ")
        try:
            cur.execute(ddl)
            print("done")
        except mysql.connector.Error as err:
            print("FAILED:", err)
    cur.close()

def load_file(conn, csv_name):
    upload_path = os.path.join(UPLOAD_DIR, csv_name).replace("\\", "\\\\")
    loader = LOAD_SQL.get(csv_name)
    if loader is None:
        print(f"[SKIP] No LOAD SQL for {csv_name}")
        return
    sql = loader(upload_path)
    cur = conn.cursor()
    try:
        print(f"Loading `{csv_name}` ...")
        cur.execute(sql)
        print(f"  -> Loaded: {csv_name}")
    except mysql.connector.Error as err:
        print(f"  [ERROR] LOAD failed for {csv_name}: {err}")
        try:
            print("  Trying LOAD DATA LOCAL INFILE fallback...")
            sql_local = sql.replace("LOAD DATA INFILE", "LOAD DATA LOCAL INFILE")
            cur.execute(sql_local)
            print(f"  -> Loaded via LOCAL: {csv_name}")
        except mysql.connector.Error as err2:
            print(f"  [FATAL] LOCAL fallback failed too: {err2}")
    finally:
        cur.close()

# -----------------------
# Main
# -----------------------
def main():
    print("\n=== MySQL Bulk Loader (fixed) ===\n")

    if not os.path.isdir(SRC_DATA_DIR):
        print(f"[ERROR] SRC_DATA_DIR not found: {SRC_DATA_DIR}")
        return
    if not os.path.isdir(UPLOAD_DIR):
        print(f"[ERROR] UPLOAD_DIR not found: {UPLOAD_DIR}")
        return

    copied = []
    for f in CSV_TABLE_MAP.keys():
        ok = copy_csv_to_upload(f)
        if ok:
            copied.append(f)
    if not copied:
        print("[ERROR] No files copied. Check your SRC_DATA_DIR and CSV filenames.")
        return

    conn = connect_db()
    print("Connected to DB.")

    try:
        cur = conn.cursor()
        cur.execute("SET SESSION sql_mode = '';")
        cur.close()
        print("[INFO] session sql_mode cleared to avoid strict insert errors.")
    except Exception as e:
        print(f"[WARN] Could not set sql_mode: {e}")

    create_tables(conn)

    load_order = [
        "users_profile_full_v3.csv",
        "monthly_expenses_12m.csv",
        "income_data_12months.csv",
        "credit_loans_12m.csv",
        "investment_data_12months.csv",
        "financial_goals_12months.csv",
        "transaction_data_12months.csv",
        "fraud_signals_12months.csv",
        # 👇 NEW
        "subscriptions_12months.csv"
    ]

    for csv_name in load_order:
        if csv_name not in copied:
            print(f"[SKIP] {csv_name} not copied earlier - skipping")
            continue
        if csv_name == "transaction_data_12months.csv":
            try:
                cur = conn.cursor()
                cur.execute("ALTER TABLE transactions DISABLE KEYS;")
                cur.close()
            except Exception:
                pass

        load_file(conn, csv_name)

        if csv_name == "transaction_data_12months.csv":
            try:
                cur = conn.cursor()
                cur.execute("ALTER TABLE transactions ENABLE KEYS;")
                cur.close()
            except Exception:
                pass

    print("\n--- Row counts ---")
    cur = conn.cursor()
    for csv_name, table in CSV_TABLE_MAP.items():
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            cnt = cur.fetchone()[0]
            print(f"{table}: {cnt:,}")
        except Exception as e:
            print(f"{table}: count failed ({e})")
    cur.close()
    conn.close()
    print("\n=== DONE ===\n")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Elapsed: {time.time() - start:.1f}s")
