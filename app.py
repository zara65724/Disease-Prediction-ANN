import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ─── Train and save model if not already saved ───────────────────────────────
def train_and_save():
    df = pd.read_csv('disease_prediction_dataset.csv')
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df[['age', 'blood_pressure', 'cholesterol', 'heart_rate', 'symptom_score']]
    y = df['disease_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
                          solver='adam', max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    pickle.dump(model,  open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    return model, scaler

# ─── Load or train ───────────────────────────────────────────────────────────
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    with st.spinner("⏳ Training model for the first time... please wait"):
        model, scaler = train_and_save()
else:
    model  = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

# ─── UI ──────────────────────────────────────────────────────────────────────
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
    prob = model.predict_proba(data)[0][1]

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
