import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('model.pkl')
    return model

model = load_model()

st.title("Sleep Disorder Prediction App")
st.write("Enter your details to predict the sleep disorder:")

# Example input widgets; ensure these match your model's expected features.
age = st.number_input("Age", min_value=20, max_value=80, value=40)
sleep_duration = st.number_input("Sleep Duration", min_value=5.0, max_value=9.0, value=7.0, step=0.1)
quality_of_sleep = st.number_input("Quality of Sleep", min_value=1, max_value=10, value=7)
physical_activity = st.number_input("Physical Activity Level", min_value=30, max_value=90, value=60)
stress_level = st.number_input("Stress Level", min_value=1, max_value=10, value=5)
heart_rate = st.number_input("Heart Rate", min_value=60, max_value=100, value=70)
daily_steps = st.number_input("Daily Steps", min_value=3000, max_value=10000, value=7000)

# Categorical inputs (update the mapping as per your encoding)
gender = st.selectbox("Gender", options=["Male", "Female"])
occupation = st.selectbox("Occupation", options=["Software Engineer", "Doctor", "Sales", "Nurse", "Other"])
bmi_category = st.selectbox("BMI Category", options=["Normal", "Overweight", "Obese", "Underweight"])
blood_pressure = st.selectbox("Blood Pressure", options=["Normal", "High"])

# Mapping the categorical variables to numerical values (update according to your encoding)
gender_code = 0 if gender == "Male" else 1
occupation_code = {"Software Engineer": 0, "Doctor": 1, "Sales": 2, "Nurse": 3, "Other": 4}.get(occupation, 4)
bmi_code = {"Normal": 0, "Overweight": 1, "Obese": 2, "Underweight": 3}.get(bmi_category, 0)
blood_pressure_code = {"Normal": 0, "High": 1}.get(blood_pressure, 0) 

# Create input DataFrame
input_df = pd.DataFrame({
    'Gender': [gender_code],
    'Age': [age],
    'Occupation': [occupation_code],
    'Sleep Duration': [sleep_duration],
    'Quality of Sleep': [quality_of_sleep],
    'Physical Activity Level': [physical_activity],
    'Stress Level': [stress_level],
    'BMI Category': [bmi_code],
    'Blood Pressure': [blood_pressure_code],
    'Heart Rate': [heart_rate],
    'Daily Steps': [daily_steps]
})

st.write(f"Input: {gender_code},{age},{occupation_code}")

if st.button("Predict Sleep Disorder"):
    prediction = model.predict(input_df)
    prediction_label = {0: "None", 1: "Sleep Apnea", 2: "Insomnia"}.get(prediction[0], "Unknown")
    st.success(f"Predicted Sleep Disorder: {prediction_label}")
