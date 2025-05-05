import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Title
st.title("📊 Student Performance Predictor")

# Load the dataset
data = pd.read_csv('Student_Performance.csv')

# Encode categorical column
label_encoder = LabelEncoder()
data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])  # Yes=1, No=0

# Features and target
X = data.drop('Performance Index', axis=1)
y = data['Performance Index']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy (R² score)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Display accuracy
st.write(f"📌 Model Accuracy (R² Score): **{r2:.2f}**")

# User Input
st.header("Enter Student Details:")

hours_studied = st.slider("Hours Studied per Day", 0, 12, 6)
previous_scores = st.slider("Previous Exam Scores", 0, 100, 70)
extracurricular = st.selectbox("Participates in Extracurricular Activities?", ["Yes", "No"])
sleep_hours = st.slider("Sleep Hours per Day", 0, 12, 7)
sample_papers = st.slider("Sample Question Papers Practiced", 0, 20, 2)

# Preprocess user input
user_input = pd.DataFrame({
    'Hours Studied': [hours_studied],
    'Previous Scores': [previous_scores],
    'Extracurricular Activities': [1 if extracurricular == "Yes" else 0],
    'Sleep Hours': [sleep_hours],
    'Sample Question Papers Practiced': [sample_papers]
})

user_input_scaled = scaler.transform(user_input)

# Predict
predicted_performance = model.predict(user_input_scaled)[0]

# Determine performance status
if predicted_performance >= 85:
    status = "⭐ Excellent"
elif predicted_performance >= 70:
    status = "👍 Good"
elif predicted_performance >= 50:
    status = "🙂 Average"
else:
    status = "⚠️ Needs Improvement"

# Output
st.success(f"🎓 Predicted Performance Index: **{predicted_performance:.2f}**")
st.info(f"📈 Performance Status: **{status}**")
# Student-Performance-Predictor
