import streamlit as st
import pandas as pd
import joblib

st.title("🪐 Exoplanet Detection Interface")
st.write("Upload data or enter feature values to see predictions.")

# load the trained model
@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

model = load_model()

# simple numeric inputs (adjust names to your real features)
age = st.number_input("Age", min_value=0, value=25)
income = st.number_input("Income", min_value=0, value=5000)
hours = st.number_input("Hours", min_value=0, value=40)
city = st.text_input("City", "Kuwait")
segment = st.text_input("Segment", "A")

if st.button("Predict"):
    X = pd.DataFrame([{
        "age": age,
        "income": income,
        "hours": hours,
        "city": city,
        "segment": segment
    }])
    pred = model.predict(X)[0]
    st.success(f"Prediction: {pred}")
