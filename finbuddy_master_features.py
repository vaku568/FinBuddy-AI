import pandas as pd
import numpy as np

# Load datasets
users = pd.read_csv("users_profile_full_v3.csv")
monthly_expenses = pd.read_csv("monthly_expenses_12m.csv")
transactions = pd.read_csv("transaction_data_12months.csv")
credit_loans = pd.read_csv("credit_loans_12m.csv")
investments = pd.read_csv("investment_data_12months.csv")
income = pd.read_csv("income_data_12months.csv")
financial_goals = pd.read_csv("financial_goals_12months.csv")
fraud_signals = pd.read_csv("fraud_signals_12months.csv")
subscriptions = pd.read_csv("subscriptions_12months.csv")

# Standardize dates based on your actual columns
monthly_expenses['month_start_date'] = pd.to_datetime(monthly_expenses['month_start_date'], errors='coerce')
transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce')
credit_loans['month'] = pd.to_datetime(credit_loans['month'], errors='coerce')
investments['month'] = pd.to_datetime(investments['month'], errors='coerce')
income['month'] = pd.to_datetime(income['month'], errors='coerce')
financial_goals['target_date'] = pd.to_datetime(financial_goals['target_date'], errors='coerce')
financial_goals['goal_created_date'] = pd.to_datetime(financial_goals['goal_created_date'], errors='coerce')
subscriptions['month'] = pd.to_datetime(subscriptions['month'], errors='coerce')
fraud_signals['date'] = pd.to_datetime(fraud_signals['date'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

# Expense categories
expense_categories = [
    'food_expense', 'groceries_expense', 'education_expense', 'subscriptions_expense',
    'fuel_expense', 'transportation_expense', 'utilities_expense', 'entertainment_expense',
    'shopping_expense', 'healthcare_expense', 'personal_care_expense', 'miscellaneous_expense'
]

# 1. Aggregate monthly spending and category percentages
monthly_expenses['total_category_expenses'] = monthly_expenses[expense_categories].sum(axis=1)
for cat in expense_categories:
    monthly_expenses[f'{cat}_pct'] = monthly_expenses[cat]/monthly_expenses['monthly_expenses'].replace({0: np.nan})

# Rolling averages and month-on-month changes
monthly_expenses = monthly_expenses.sort_values(by=['user_id', 'month_start_date'])
for cat in expense_categories + ['monthly_expenses', 'monthly_surplus', 'savings_rate']:
    monthly_expenses[f'{cat}_1m_avg'] = monthly_expenses.groupby('user_id')[cat].transform(lambda x: x.rolling(window=1).mean())
    monthly_expenses[f'{cat}_3m_avg'] = monthly_expenses.groupby('user_id')[cat].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    monthly_expenses[f'{cat}_6m_avg'] = monthly_expenses.groupby('user_id')[cat].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    monthly_expenses[f'{cat}_mom_change'] = monthly_expenses.groupby('user_id')[cat].pct_change()

# 2. Risk assessment aggregation
fraud_by_user = fraud_signals.groupby('user_id').agg({
    'fraud_label': lambda x: x.nunique(),
    'flagged_amount': 'sum',
    'severity': lambda x: x.mode()[0] if not x.mode().empty else 'low'
}).rename(columns={'fraud_label': 'fraud_types_count', 'flagged_amount': 'total_fraud_flagged_amount'})

credit_loans_avg = credit_loans.groupby('user_id').agg({
    'credit_utilization': 'mean',
    'loan_to_income_ratio': 'mean',
    'outstanding_credit': 'mean'
})

risk_features = users.set_index('user_id').join(fraud_by_user).join(credit_loans_avg)
risk_features.fillna(0, inplace=True)

risk_features['composite_risk_score'] = (
    0.4*risk_features['credit_utilization'] +
    0.3*risk_features['loan_to_income_ratio'] +
    0.2*(risk_features['fraud_types_count'] / (risk_features['fraud_types_count'].max() + 1)) +
    0.1*(risk_features['total_fraud_flagged_amount'] / (risk_features['total_fraud_flagged_amount'].max() + 1))
)

# 3. Savings potential from user and investment data
investment_summary = investments.groupby('user_id').agg({
    'total_investment_value': 'mean',
    'stocks': 'mean',
    'sip': 'mean',
    'crypto': 'mean',
    'gold_bonds': 'mean'
})

savings_features = users.set_index('user_id')[['savings_rate', 'monthly_surplus', 'investment_amount', 'has_investments']].join(investment_summary)
savings_features.fillna(0, inplace=True)

# 4. Investment clustering input - diversification index
investment_summary['diversification_index'] = investment_summary[['stocks', 'sip', 'crypto', 'gold_bonds']].gt(0).sum(axis=1)

# 5. Subscription churn features
subscription_features = subscriptions.groupby('user_id').agg({
    'churn_flag': 'mean',
    'churn_rate': 'mean',
    'active_subs': 'mean',
    'canceled_subs': 'mean',
    'subs_to_fee_ratio': 'mean'
})

# 6. Seasonal spending features - compute distribution of monthly expenditure
seasonal_spend = monthly_expenses.groupby('user_id').agg({cat: 'mean' for cat in expense_categories})
seasonal_spend['spend_std_dev'] = monthly_expenses.groupby('user_id')['monthly_expenses'].std()
seasonal_spend['spend_mean'] = monthly_expenses.groupby('user_id')['monthly_expenses'].mean()
seasonal_spend['seasonality_index'] = seasonal_spend['spend_std_dev'] / seasonal_spend['spend_mean']

# 7. Life event detection features based on financial goals
goal_features = financial_goals.groupby('user_id').agg({
    'progress_percent': 'mean',
    'priority_score': 'max',
    'months_to_target': 'mean',
    'target_amount': 'sum'
})

# 8. Cashflow features
cashflow_features = monthly_expenses.groupby('user_id').agg({
    'monthly_surplus': ['mean', 'std'],
    'savings_rate': ['mean']
})
cashflow_features.columns = ['cashflow_surplus_mean', 'cashflow_surplus_std', 'avg_savings_rate']

# 9. Merchant behavior features
merchant_spend = transactions.groupby(['user_id', 'category'])['amount'].sum().unstack(fill_value=0)
merchant_spend_pct = merchant_spend.div(merchant_spend.sum(axis=1), axis=0)

# 10. Goal achievement features (aggregated)
goal_achievement_features = financial_goals.groupby('user_id').agg({
    'progress_percent': 'mean',
    'priority_score': 'max'
})

# 11. Archetype from users dataset
archetype_features = users.set_index('user_id')[['user_archetype']]

# Combine all feature sets
feature_dfs = [
    risk_features[['composite_risk_score']],
    savings_features,
    investment_summary,
    subscription_features,
    seasonal_spend,
    goal_features,
    cashflow_features,
    goal_achievement_features,
    archetype_features
]

master_features = pd.concat(feature_dfs, axis=1).fillna(0)

master_features.to_csv("finbuddy_master_features.csv")

print(f"Feature engineering complete: {master_features.shape}")
