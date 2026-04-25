import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model  = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Disease Prediction System", layout="centered")

st.title("🧠 ANN Disease Prediction System")
st.write("Enter patient details to predict disease risk level")

age     = st.number_input("Age", 1, 120, 25)
bp      = st.number_input("Blood Pressure", 50, 200, 120)
chol    = st.number_input("Cholesterol", 100, 400, 200)
hr      = st.number_input("Heart Rate", 40, 180, 80)
symptom = st.slider("Symptom Score", 0, 10, 5)

def predict_disease(age, bp, chol, hr, symptom):
    data = np.array([[age, bp, chol, hr, symptom]])
    data = scaler.transform(data)
    prob = model.predict_proba(data)[0][1]   # probability of class 1

    if prob < 0.3:
        return "Low Risk 🟢", prob
    elif prob < 0.7:
        return "Medium Risk 🟡", prob
    else:
        return "High Risk 🔴", prob

if st.button("Predict"):
    result, prob = predict_disease(age, bp, chol, hr, symptom)
    st.success(f"Prediction: {result}")
    st.info(f"Risk Probability: {prob:.2%}")
