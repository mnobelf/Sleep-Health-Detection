import streamlit as st
import pandas as pd
import joblib

# Load your saved model
@st.cache_data
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Sleep Disorder Prediction App")
st.write("Enter your details to predict the sleep disorder:")

########################################
# 1. User Input
########################################

# Numeric inputs
age = st.number_input("Age", min_value=20, max_value=80, value=40)
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=5.0, max_value=9.0, value=7.0, step=0.1)
quality_of_sleep = st.number_input("Quality of Sleep (1-10)", min_value=1, max_value=10, value=7)
physical_activity = st.number_input("Physical Activity Level (mins/day)", min_value=30, max_value=90, value=60)
stress_level = st.number_input("Stress Level (1-10)", min_value=1, max_value=10, value=5)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=100, value=70)
daily_steps = st.number_input("Daily Steps", min_value=3000, max_value=12000, value=7000)

# Categorical inputs (assuming you have label encoding for these)
gender = st.selectbox("Gender", options=["Male", "Female"])
occupation = st.selectbox("Occupation", options=["Accountant","Doctor","Engineer","Lawyer","Manager","Nurse","Sales Representative","Salesperson","Scientist","Software Engineer","Teacher"])
bmi_category = st.selectbox("BMI Category", options=["Normal", "Underweight", "Obese", "Overweight"])

########################################
# 2. Binning Functions
########################################

def bin_age(val):
    # Bins: (26.968, 43.0], (43.0, 59.0]
    if val <= 43.0:
        return 0
    else:
        return 1

def bin_sleep_duration(val):
    # Bins: (5.76, 6.7], (6.7, 7.6], (7.6, 8.5]
    if val <= 6.7:
        return 0
    elif val <= 7.6:
        return 1
    else:
        return 2

def bin_physical_activity(val):
    # Bins: (29.97, 44.5], (44.5, 59.0], (59.0, 73.5], (73.5, 88.0]
    if val <= 45:
        return 0
    elif val <= 60:
        return 1
    elif val <= 75:
        return 2
    else:
        return 3

def bin_heart_rate(val):
    # Bins: (64.979, 70.25], (70.25, 75.5], (75.5, 80.75], (80.75, 86.0]
    if val <= 70.25:
        return 0
    elif val <= 75.5:
        return 1
    elif val <= 80.75:
        return 2
    else:
        return 3

def bin_daily_steps(val):
    # Bins: (2990.0, 5249.0], (5249.0, 7498.0], (7498.0, 9747.0], (9747.0, 11996.0]
    if val <= 4750.0:
        return 0
    elif val <= 6500.0:
        return 1
    elif val <= 8250.0:
        return 2
    else:
        return 3

########################################
# 3. Map Categorical Inputs to Encoded Values
########################################

# These mappings should match how you label-encoded them in your training code
gender_map = {"Male": 1, "Female": 0}
occupation_map = {
    "Accountant": 0,
    "Doctor": 1,
    "Engineer": 2,
    "Lawyer": 3,
    "Manager": 4,
    "Nurse": 5,
    "Sales Representative": 6,
    "Salesperson": 7,
    "Scientist": 8,
    "Software Engineer": 9,
    "Teacher": 10
}
bmi_map = {"Normal": 0, "Underweight": 1, "Obese": 2, "overweight": 3}

# If your notebook had blood pressure binned or encoded, do similarly here
blood_pressure_code = 0  # or any logic needed

########################################
# 4. Convert Raw Inputs -> Binned/Encoded -> DataFrame
########################################

binned_age = bin_age(age)
binned_sleep = bin_sleep_duration(sleep_duration)
binned_physical = bin_physical_activity(physical_activity)
binned_heart = bin_heart_rate(heart_rate)
binned_steps = bin_daily_steps(daily_steps)

input_df = pd.DataFrame({
    'Gender': [gender_map[gender]],
    'Age': [binned_age],
    'Occupation': [occupation_map[occupation]],
    'Sleep Duration': [binned_sleep],
    'Quality of Sleep': [quality_of_sleep],
    'Physical Activity Level': [binned_physical],
    'Stress Level': [stress_level],
    'BMI Category': [bmi_map[bmi_category]],
    'Blood Pressure': [blood_pressure_code],
    'Heart Rate': [binned_heart],
    'Daily Steps': [binned_steps]
})

########################################
# 5. Predict
########################################

if st.button("Predict Sleep Disorder"):
    prediction = model.predict(input_df)
    # Map prediction back to labels
    # (Ensure these match your training label encoding for "Sleep Disorder")
    prediction_label = {0: "Insomnia", 1: "None", 2: "Sleep Apnea"}.get(prediction[0], "Unknown")
    st.success(f"Predicted Sleep Disorder: {prediction_label}")
