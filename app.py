
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Disease Prediction System", layout="centered")

st.title("🧠 ANN Disease Prediction System")
st.write("Enter patient details to predict disease risk level")

age = st.number_input("Age", 1, 120, 25)
bp = st.number_input("Blood Pressure", 50, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
hr = st.number_input("Heart Rate", 40, 180, 80)
symptom = st.slider("Symptom Score", 0, 10, 5)

def predict_disease(age, bp, chol, hr, symptom):
    data = np.array([[age, bp, chol, hr, symptom]])
    data = scaler.transform(data)
    pred = model.predict(data)[0][0]

    if pred < 0.3:
        return "Low Risk 🟢"
    elif pred < 0.7:
        return "Medium Risk 🟡"
    else:
        return "High Risk 🔴"

if st.button("Predict"):
    result = predict_disease(age, bp, chol, hr, symptom)
    st.success(f"Prediction: {result}")
