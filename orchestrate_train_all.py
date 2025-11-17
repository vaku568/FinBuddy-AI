import subprocess

def run_script(script_name):
    print(f"Running {script_name} ...")
    ret = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(ret.stdout)
    if ret.returncode != 0:
        print(f"Error in {script_name}:")
        print(ret.stderr)
        exit(1)

if __name__ == "__main__":
    scripts = [
        "future_spending.py",
        "category_forecast.py",
        "risk_assessment.py",
        "savings_potential.py",
        "investment_clustering.py",
        "subscription_churn.py",
        "seasonal_spending.py",
        "life_event_detection.py",
        "cashflow_liquidity.py",
        "merchant_behavior.py",
        "goal_achievement.py",
        "archetype_classifier.py"
    ]
    for script in scripts:
        run_script(script)

    print("All models trained successfully.")
