import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ─── Train and save model ─────────────────────────────────────────────────────
def train_and_save():
    df = pd.read_csv('disease_prediction_dataset.csv')
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df[['age', 'blood_pressure', 'cholesterol', 'heart_rate', 'symptom_score']]
    y = df['disease_label']  # "Low Risk", "Medium Risk", "High Risk"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    pickle.dump(model,  open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    return model, scaler, acc

# ─── Load or train ────────────────────────────────────────────────────────────
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    with st.spinner("⏳ Training model for the first time... please wait"):
        model, scaler, acc = train_and_save()
    st.success(f"✅ Model trained! Accuracy: {acc:.2%}")
else:
    model  = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

# ─── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disease Prediction System", layout="centered")

st.title("🧠 ANN Disease Prediction System")
st.write("Enter patient details to predict disease risk level")

col1, col2 = st.columns(2)
with col1:
    age  = st.number_input("Age", 18, 79, 40)
    bp   = st.number_input("Blood Pressure", 50, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
with col2:
    hr      = st.number_input("Heart Rate", 60, 119, 90)
    symptom = st.slider("Symptom Score", 0, 9, 4)

EMOJI = {
    "Low Risk":    "🟢",
    "Medium Risk": "🟡",
    "High Risk":   "🔴",
}
COLOR = {
    "Low Risk":    "green",
    "Medium Risk": "orange",
    "High Risk":   "red",
}

if st.button("🔍 Predict", use_container_width=True):
    data  = scaler.transform(np.array([[age, bp, chol, hr, symptom]]))
    label = model.predict(data)[0]               # "Low Risk" / "Medium Risk" / "High Risk"
    probs = model.predict_proba(data)[0]
    classes = model.classes_

    emoji = EMOJI.get(label, "❓")
    st.markdown(f"### Prediction: {emoji} **{label}**")

    st.write("**Class Probabilities:**")
    for cls, prob in zip(classes, probs):
        st.progress(float(prob), text=f"{EMOJI.get(cls,'')} {cls}: {prob:.1%}")
