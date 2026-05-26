import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL AND SCALER
# =========================

model = joblib.load("../models/random_forest.pkl")

scaler = joblib.load("../models/scaler.pkl")

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("📊 Employee Attrition Prediction System")

st.write(
    "Predict whether an employee is likely to leave the company."
)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("About")

st.sidebar.info(
    """
    This machine learning application predicts employee attrition
    using important employee-related features.
    """
)

# =========================
# USER INPUTS
# =========================

st.subheader("Enter Employee Details")

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=200000,
    value=5000
)

overtime = st.selectbox(
    "OverTime",
    ["No", "Yes"]
)

age = st.slider(
    "Age",
    18,
    60,
    30
)

total_working_years = st.slider(
    "Total Working Years",
    0,
    40,
    5
)

years_at_company = st.slider(
    "Years At Company",
    0,
    40,
    3
)

distance_from_home = st.slider(
    "Distance From Home",
    1,
    30,
    5
)

# =========================
# ENCODE INPUT
# =========================

overtime = 1 if overtime == "Yes" else 0

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Attrition"):

    # CREATE DATAFRAME

    input_data = pd.DataFrame({
        'MonthlyIncome': [monthly_income],
        'OverTime': [overtime],
        'Age': [age],
        'TotalWorkingYears': [total_working_years],
        'YearsAtCompany': [years_at_company],
        'DistanceFromHome': [distance_from_home]
    })

    # SCALE INPUT

    input_scaled = scaler.transform(input_data)

    # PREDICTION

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    confidence = round(max(probability[0]) * 100, 2)

    # RESULT

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error(
            f"⚠️ Employee is likely to leave the company.\n\n"
            f"Confidence: {confidence}%"
        )

    else:

        st.success(
            f"✅ Employee is likely to stay in the company.\n\n"
            f"Confidence: {confidence}%"
        )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Employee Attrition Prediction using Machine Learning"
)