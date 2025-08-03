import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Attack Risk Predictor")
st.markdown("Enter the following values to assess your risk:")

# Input features
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", [0, 1])  # 1 = male, 0 = female
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced", min_value=0.0, max_value=6.0, step=0.1)
slope = st.selectbox("Slope of ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)", [1, 2, 3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict_proba(input_scaled)[0][1]

    if prediction >= 0.7:
        st.error(f"High Risk ({prediction*100:.2f}%) – Please consult a doctor.")
    elif prediction >= 0.4:
        st.warning(f"Moderate Risk ({prediction*100:.2f}%) – Take precautions.")
    else:
        st.success(f"Low Risk ({prediction*100:.2f}%) – Keep maintaining a healthy lifestyle.")
