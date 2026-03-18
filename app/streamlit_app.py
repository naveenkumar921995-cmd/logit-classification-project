import streamlit as st
import numpy as np
import pickle

# Load model
model, sc = pickle.load(open('models/logistic_model.pkl', 'rb'))

st.title("Logistic Regression Predictor")

# Inputs
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")

if st.button("Predict"):
    data = np.array([[feature1, feature2]])
    data = sc.transform(data)
    result = model.predict(data)
    st.success(f"Prediction: {result[0]}")
