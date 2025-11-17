# user_profile_full_generator_v3.py
# Generates 10,000 realistic user profiles (age 16-65) with consistent education/is_student values
# Output: users_profile_full_v3.csv

import numpy as np
import pandas as pd
import random
from scipy import stats

class FullUserProfileGeneratorV3:
    def __init__(self, num_users=10000, save_path="users_profile_full_v3.csv", seed=42):
        self.num_users = num_users
        self.save_path = save_path
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Age distribution parameters will be used to draw ages in the 16-65 range
        self.age_loc = 28
        self.age_scale = 10

        # Income base distribution (log-normal) - we'll multiply by education/age multipliers
        self.base_income_dist = stats.lognorm(s=0.9, scale=30000)

        # Archetypes (for behavioral realism)
        self.archetypes = [
            "conservative_saver",
            "balanced_planner",
            "aggressive_investor",
            "impulsive_spender",
            "goal_oriented_optimizer"
        ]

        # Education levels - ordered (lowest -> highest)
        self.education_levels = ["high_school", "bachelors", "masters", "professional", "phd"]

        # Communication / behavior options
        self.money_approach = ['meticulous_tracker', 'rough_idea', 'struggle_control', 'avoidant']
        self.decision_style = ['research_heavy', 'peer_influenced', 'emotional', 'quick_decision']
        self.goal_style = ['clear_goals', 'flexible_goals', 'short_term_focus', 'uncertain']
        self.communication_pref = ['app_only', 'sms', 'email', 'call']
        self.info_processing = ['visual', 'textual', 'mixed']

        # Cities and metro flag
        self.metro_cities = ['Mumbai','Delhi','Bangalore','Hyderabad','Chennai','Pune']
        self.non_metro_cities = ['Ahmedabad','Surat','Jaipur','Lucknow','Nagpur','Kochi','Indore']

    def _draw_age(self):
        """Draw an age between 16 and 65 with a center around young adults."""
        # Use a rounded normal with clipping
        age = int(round(np.random.normal(loc=self.age_loc, scale=self.age_scale)))
        age = max(16, min(65, age))
        return age

    def _assign_education_by_age(self, age):
        """
        Assign education level probabilistically depending on age:
        - 16-17  : mostly high_school
        - 18-22  : high_school or bachelors (students)
        - 23-26  : bachelors / masters / early professionals
        - 27-35  : bachelors/masters/professional
        - 36+    : professional / masters / some phd
        """
        if age <= 17:
            return "high_school"
        elif 18 <= age <= 22:
            # Many will be bachelors students or recent grads
            return np.random.choice(["high_school","bachelors"], p=[0.15, 0.85])
        elif 23 <= age <= 26:
            return np.random.choice(["bachelors","masters","professional"], p=[0.65,0.25,0.10])
        elif 27 <= age <= 35:
            return np.random.choice(["bachelors","masters","professional"], p=[0.40,0.40,0.20])
        elif 36 <= age <= 50:
            return np.random.choice(["bachelors","masters","professional","phd"], p=[0.35,0.40,0.20,0.05])
        else:
            # 51-65
            return np.random.choice(["bachelors","masters","professional","phd"], p=[0.45,0.35,0.15,0.05])

    def _is_current_student(self, age, education):
        """
        Determine is_student realistically:
        - High chance if age between 16-24 and education indicates 'bachelors' or 'high_school'
        - Lower chance for masters/PhD students in typical age ranges
        """
        if education == "high_school":
            return np.random.choice([True, False], p=[0.7, 0.3]) if age <= 19 else False
        if education == "bachelors":
            # most 18-24 with bachelors are still students or recent grads
            if age <= 23:
                return np.random.choice([True, False], p=[0.75, 0.25])
            elif 24 <= age <= 27:
                return np.random.choice([True, False], p=[0.2, 0.8])
            else:
                return False
        if education == "masters":
            # masters students often are 22-30
            if 22 <= age <= 30:
                return np.random.choice([True, False], p=[0.35, 0.65])
            else:
                return False
        if education == "professional":
            return False
        if education == "phd":
            # PhD students sometimes in 25-35
            if 25 <= age <= 40:
                return np.random.choice([True, False], p=[0.30, 0.70])
            return False
        return False

    def _graduation_age(self, education):
        """
        Typical finishing (graduation) ages per education:
         - high_school: 17-18
         - bachelors: ~21-23
         - masters: ~23-25
         - professional: ~25-28
         - phd: ~28-35
        """
        if education == "high_school":
            return 17
        if education == "bachelors":
            return 22
        if education == "masters":
            return 24
        if education == "professional":
            return 26
        if education == "phd":
            return 30
        return 22

    def _years_experience(self, age, is_student, education):
        """Estimate years of professional experience consistent with education & student status."""
        if is_student:
            return 0
        grad_age = self._graduation_age(education)
        # some people start working before finishing degree; allow small positive experience if age > grad_age
        possible_years = max(0, age - grad_age)
        # add some variation but keep sensible
        years = int(max(0, np.random.poisson(lam=max(0.5, possible_years * 0.6))))
        # cap it reasonably
        years = min(years, age - 16)
        return years

    def _monthly_income(self, age, education, years_experience, is_student, city_is_metro):
        """
        Generate monthly income with dependence on:
        - education (higher education -> higher multiplier)
        - years_experience (linear factor)
        - student -> low income probability (internships / part-time)
        - metro -> higher income multiplier
        """
        # base draw
        base = int(self.base_income_dist.rvs())

        # education multiplier
        edu_multiplier = {
            "high_school": 0.65,
            "bachelors": 1.0,
            "masters": 1.25,
            "professional": 1.45,
            "phd": 1.6
        }[education]

        # experience factor
        exp_factor = 1 + (years_experience * 0.05)

        # student discount
        student_factor = 0.35 if is_student else 1.0

        # metro uplift
        metro_factor = 1.25 if city_is_metro else 1.0

        income = int(max(5000, base * edu_multiplier * exp_factor * student_factor * metro_factor))
        # add some noise
        income = int(income * np.random.uniform(0.85, 1.15))
        return income

    def _spend_allocation(self, monthly_expenses, archetype, age):
        """
        Allocate monthly expenses into categories.
        We slightly adjust proportions by archetype & age group.
        Returns dict with category amounts summing (approximately) to monthly_expenses.
        """
        # baseline split
        split = {
            'food_expense': 0.20,
            'groceries_expense': 0.14,
            'education_expense': 0.06,
            'subscriptions_expense': 0.05,
            'fuel_expense': 0.05,
            'transportation_expense': 0.10,
            'utilities_expense': 0.07,
            'entertainment_expense': 0.12,
            'shopping_expense': 0.09,
            'healthcare_expense': 0.03,
            'personal_care_expense': 0.04,
            'miscellaneous_expense': 0.05,
        }

        # archetype adjustments
        if archetype == "conservative_saver":
            split['savings_boost'] = 0  # informational
            split['entertainment_expense'] *= 0.6
            split['shopping_expense'] *= 0.6
            split['groceries_expense'] *= 1.1
        elif archetype == "impulsive_spender":
            split['entertainment_expense'] *= 1.3
            split['shopping_expense'] *= 1.4
            split['food_expense'] *= 1.1
            split['groceries_expense'] *= 0.9
        elif archetype == "aggressive_investor":
            split['education_expense'] *= 1.2
            split['savings_boost'] = 0
            split['shopping_expense'] *= 0.8
        elif archetype == "goal_oriented_optimizer":
            split['savings_boost'] = 0
            split['education_expense'] *= 1.1
            split['subscriptions_expense'] *= 0.9

        # age effects (younger -> more entertainment, older -> more utilities/healthcare)
        if age <= 25:
            split['entertainment_expense'] *= 1.2
            split['subscriptions_expense'] *= 1.2
        elif age >= 50:
            split['healthcare_expense'] *= 1.6
            split['utilities_expense'] *= 1.1

        # normalize to sum to 1 (ignore any temporary keys)
        keys = [k for k in split.keys() if k.endswith('_expense')]
        vals = np.array([split[k] for k in keys], dtype=float)
        vals = vals / vals.sum()

        alloc = {k: int(round(v * monthly_expenses)) for k, v in zip(keys, vals)}

        # small correction to ensure sum matches monthly_expenses (due to rounding)
        diff = monthly_expenses - sum(alloc.values())
        if diff != 0:
            # add diff to the largest category
            largest = max(alloc, key=alloc.get)
            alloc[largest] += diff

        return alloc

    def _yes_no(self, cond):
        return "Yes" if cond else "No"

    def generate(self):
        records = []
        for uid in range(self.num_users):
            age = self._draw_age()
            education = self._assign_education_by_age(age)
            is_student_bool = self._is_current_student(age, education)
            # choose city
            city = (np.random.choice(self.metro_cities + self.non_metro_cities))
            is_metro_bool = city in self.metro_cities

            # years experience and income depend on education and student flag
            years_experience = self._years_experience(age, is_student_bool, education)
            monthly_income = self._monthly_income(age, education, years_experience, is_student_bool, is_metro_bool)

            # expense ratio depends on age/archetype; pick archetype
            archetype = np.random.choice(self.archetypes)
            # younger / students often spend higher share; older often save more (modest effect)
            if is_student_bool:
                expense_ratio = np.clip(np.random.normal(0.85, 0.06), 0.6, 0.98)
            elif age <= 25:
                expense_ratio = np.clip(np.random.normal(0.78, 0.08), 0.5, 0.98)
            elif age <= 35:
                expense_ratio = np.clip(np.random.normal(0.72, 0.08), 0.45, 0.95)
            else:
                expense_ratio = np.clip(np.random.normal(0.68, 0.08), 0.35, 0.95)

            monthly_expenses = int(max(0, monthly_income * expense_ratio))
            monthly_surplus = monthly_income - monthly_expenses
            savings_rate = round(monthly_surplus / monthly_income if monthly_income > 0 else 0.0, 3)

            has_investments = bool((monthly_surplus > 2500) and (np.random.random() < 0.6))
            investment_amount = int(round(monthly_surplus * np.random.uniform(0.1, 0.4))) if has_investments else 0
            debt_to_income = round(np.random.uniform(0.05, 0.6), 2)

            # create spending allocation
            allocation = self._spend_allocation(monthly_expenses, archetype, age)

            rec = {
                "user_id": uid,
                "age": age,
                "month": None,
                "monthly_income": monthly_income,
                "education": education,
                "city": city,
                "is_metro": self._yes_no(is_metro_bool),
                "dependents": 0 if is_student_bool else np.random.randint(0, 3),
                "years_experience": years_experience,
                "is_student": self._yes_no(is_student_bool),
                "risk_tolerance": round(np.clip(np.random.normal(3.0, 0.8), 1.0, 5.0), 1),
                "monthly_expenses": monthly_expenses,
                # category spends:
                "food_expense": allocation.get('food_expense', 0),
                "groceries_expense": allocation.get('groceries_expense', 0),
                "education_expense": allocation.get('education_expense', 0),
                "subscriptions_expense": allocation.get('subscriptions_expense', 0),
                "fuel_expense": allocation.get('fuel_expense', 0),
                "transportation_expense": allocation.get('transportation_expense', 0),
                "utilities_expense": allocation.get('utilities_expense', 0),
                "entertainment_expense": allocation.get('entertainment_expense', 0),
                "shopping_expense": allocation.get('shopping_expense', 0),
                "healthcare_expense": allocation.get('healthcare_expense', 0),
                "personal_care_expense": allocation.get('personal_care_expense', 0),
                "miscellaneous_expense": allocation.get('miscellaneous_expense', 0),
                "monthly_surplus": monthly_surplus,
                "savings_rate": savings_rate,
                "investment_amount": investment_amount,
                "has_investments": self._yes_no(has_investments),
                "technology_comfort": round(max(1.0, min(5.0, 6 - (age / 16.0))), 1),
                "money_management_approach": np.random.choice(self.money_approach),
                "decision_making_style": np.random.choice(self.decision_style),
                "goal_setting_behavior": np.random.choice(self.goal_style),
                "preferred_communication": np.random.choice(self.communication_pref),
                "information_processing": np.random.choice(self.info_processing),
                "user_archetype": archetype,
                "debt_to_income": debt_to_income
            }
            records.append(rec)

        df = pd.DataFrame.from_records(records)

        # final safety checks: ensure numeric columns are ints and no negative values
        int_cols = [
            'user_id','age','monthly_income','dependents','years_experience',
            'monthly_expenses','food_expense','groceries_expense','education_expense',
            'subscriptions_expense','fuel_expense','transportation_expense','utilities_expense',
            'entertainment_expense','shopping_expense','healthcare_expense','personal_care_expense',
            'miscellaneous_expense','monthly_surplus','investment_amount'
        ]
        for c in int_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(int)
                df[c] = df[c].clip(lower=0)

        # Save
        df.to_csv(self.save_path, index=False)
        print(f"✅ Generated {len(df)} user profiles → {self.save_path}")
        print(f"Columns: {df.columns.tolist()}")
        return df


if __name__ == "__main__":
    gen = FullUserProfileGeneratorV3(num_users=10000)
    df = gen.generate()
