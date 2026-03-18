import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer

MODEL_PATH = "models/logistic_model.pkl"

# Train model if not exists
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training new model...")

    dataset = pd.read_csv("data/logit_classification.csv")
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, -1].values

    sc = Normalizer()
    X = sc.fit_transform(X)

    model = LogisticRegression()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, sc), f)

    st.success("Model trained & saved successfully ✅")

# Load model safely
with open(MODEL_PATH, "rb") as f:
    model, sc = pickle.load(f)
