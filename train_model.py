import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('disease_prediction_dataset.csv')

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features & label
X = df[['age', 'blood_pressure', 'cholesterol', 'heart_rate', 'symptom_score']]
y = df['disease_label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Build ANN using sklearn (no TensorFlow needed!)
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {acc:.4f}")

# Save both model and scaler
pickle.dump(model,  open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved!")
