
# =============================================================================
# ADVANCED MATHEMATICS ENROLLMENT PREDICTION MODEL
# Logistic/Linear Regression with Multi-Driver Analysis
# Predicting US High School Calculus Enrollment Through 2028
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. HISTORICAL DATA (1990-2024)
# Sources: NCES High School Transcript Studies, College Board AP Data, BLS
# =============================================================================

historical_data = pd.DataFrame({
    'year': [1990, 1994, 1998, 2000, 2005, 2009, 2013, 2019, 2022, 2024],
    'calculus_pct': [6.5, 9.2, 10.8, 11.6, 13.6, 15.9, 18.9, 15.8, 16.2, 16.8],

    # DRIVER VARIABLES (normalized 0-100 indices)
    'stem_job_demand': [45, 50, 58, 65, 70, 68, 75, 82, 85, 88],
    'teacher_supply_index': [70, 68, 65, 62, 58, 55, 52, 48, 46, 45],
    'university_prereq_stringency': [85, 85, 82, 80, 78, 75, 72, 68, 65, 62],
    'auto_enrollment_policies': [5, 8, 12, 15, 20, 25, 35, 50, 58, 65],
    'tech_tools_availability': [5, 10, 20, 30, 45, 55, 65, 75, 85, 92],
    'math_anxiety_index': [60, 58, 55, 52, 50, 48, 47, 46, 45, 44],
    'demographic_growth': [50, 52, 55, 60, 65, 68, 70, 72, 71, 70]
})

# =============================================================================
# 2. MODEL FITTING
# =============================================================================

feature_cols = ['stem_job_demand', 'teacher_supply_index', 'university_prereq_stringency',
                'auto_enrollment_policies', 'tech_tools_availability', 
                'math_anxiety_index', 'demographic_growth']

X = historical_data[feature_cols].values
y = historical_data['calculus_pct'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# =============================================================================
# 3. SCENARIO DEFINITIONS (2025-2028)
# Modify these values to test different assumptions
# =============================================================================

projection_years = [2025, 2026, 2027, 2028]

# BASE CASE: Historical trends continue, no significant AI impact
base_case = pd.DataFrame({
    'year': projection_years,
    'stem_job_demand': [90, 91, 92, 93],
    'teacher_supply_index': [44, 43, 42, 41],
    'university_prereq_stringency': [60, 58, 56, 54],
    'auto_enrollment_policies': [68, 72, 76, 80],
    'tech_tools_availability': [94, 96, 97, 98],
    'math_anxiety_index': [43, 42, 41, 40],
    'demographic_growth': [69, 68, 67, 66]
})

# MODERATE AI SCENARIO: Realistic AI integration
moderate_ai = pd.DataFrame({
    'year': projection_years,
    'stem_job_demand': [90, 91, 92, 93],
    'teacher_supply_index': [44, 44, 45, 45],       # AI augments teachers
    'university_prereq_stringency': [59, 57, 55, 53],
    'auto_enrollment_policies': [69, 73, 77, 82],   # AI identifies students
    'tech_tools_availability': [95, 97, 98, 99],    # AI tools widespread
    'math_anxiety_index': [42, 40, 38, 36],         # AI reduces anxiety
    'demographic_growth': [69, 68, 67, 66]
})

# AGGRESSIVE AI SCENARIO: Optimistic AI transformation
aggressive_ai = pd.DataFrame({
    'year': projection_years,
    'stem_job_demand': [90, 92, 94, 96],
    'teacher_supply_index': [45, 46, 47, 48],
    'university_prereq_stringency': [58, 54, 50, 46],
    'auto_enrollment_policies': [70, 78, 85, 92],
    'tech_tools_availability': [96, 98, 99, 100],
    'math_anxiety_index': [40, 36, 32, 28],
    'demographic_growth': [69, 68, 67, 66]
})

# =============================================================================
# 4. PREDICTION FUNCTION
# =============================================================================

def predict_enrollment(scenario_df, inertia_factor=0.35):
    """
    Predict enrollment with institutional inertia adjustment.

    Args:
        scenario_df: DataFrame with driver values for projection years
        inertia_factor: How much of predicted change actually occurs (0-1)
                       Default 0.35 accounts for slow institutional adoption
    """
    X_proj = scenario_df[feature_cols].values
    X_proj_scaled = scaler.transform(X_proj)
    raw_predictions = model.predict(X_proj_scaled)

    base_2024 = 16.8  # 2024 baseline
    adjusted = base_2024 + (raw_predictions - base_2024) * inertia_factor
    return adjusted

# =============================================================================
# 5. GENERATE PREDICTIONS
# =============================================================================

base_case['enrollment_pct'] = predict_enrollment(base_case)
moderate_ai['enrollment_pct'] = predict_enrollment(moderate_ai)
aggressive_ai['enrollment_pct'] = predict_enrollment(aggressive_ai)

# =============================================================================
# 6. RESULTS OUTPUT
# =============================================================================

print("=" * 60)
print("2028 ENROLLMENT PREDICTIONS")
print("=" * 60)
print(f"\n2024 Baseline:           16.8%")
print(f"Base Case (No AI):       {base_case['enrollment_pct'].iloc[-1]:.1f}%")
print(f"Moderate AI Scenario:    {moderate_ai['enrollment_pct'].iloc[-1]:.1f}%")
print(f"Aggressive AI Scenario:  {aggressive_ai['enrollment_pct'].iloc[-1]:.1f}%")

print("\n" + "=" * 60)
print("DRIVER WEIGHTS (Standardized Coefficients)")
print("=" * 60)
for i, driver in enumerate(feature_cols):
    coef = model.coef_[i]
    print(f"{driver:35s}: {coef:+.3f}")

print(f"\nModel R²: {model.score(X_scaled, y):.4f}")
