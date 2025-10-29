# heart_disease/heart_app.py

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("heart_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction")

# Input fields
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250)
exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", [1, 0])
oldpeak = st.number_input("ST depression", 0.0, 10.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = normal; 1 = fixed; 2 = reversible)", [0, 1, 2])

# Make prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    result = "Positive (At Risk)" if prediction[0][0] > 0.5 else "Negative (No Risk)"
    st.subheader(f"Prediction: {result}")
