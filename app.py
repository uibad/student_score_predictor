import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Student Score Predictor", page_icon="📊", layout="centered")

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("student_score_model.pkl")

model = load_model()

# ---------------------------
# UI
# ---------------------------
st.title("📊 Student Score Predictor")
st.write("Study Hours ke basis par predicted Score dekhne ke liye neeche hours darj karein.")

hours = st.number_input("Study Hours:", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

if st.button("Predict Score"):
    prediction = model.predict(np.array([[hours]]))[0]
    st.success(f"Estimated Score: **{prediction:.2f}**")

st.caption("Model: RandomForestRegressor (pretrained, student_score_model.pkl) | Built with Streamlit")
