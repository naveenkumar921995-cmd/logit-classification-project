import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("📊 Logistic Regression Classification App")

# ================= TRAIN MODEL =================

@st.cache_resource
def train_model():
    dataset = pd.read_csv("data/logit_classification.csv")

    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    sc = Normalizer()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    return model, sc, cm, accuracy, train_acc, test_acc


model, sc, cm, accuracy, train_acc, test_acc = train_model()

# ================= SHOW METRICS =================

st.subheader("📈 Model Performance")

st.write("Confusion Matrix:")
st.write(cm)

st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Training Accuracy (Bias): {train_acc:.2f}")
st.write(f"Testing Accuracy (Variance): {test_acc:.2f}")

# ================= USER INPUT =================

st.subheader("🔮 Single Prediction")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    data = np.array([[f1, f2]])
    data = sc.transform(data)
    result = model.predict(data)
    st.success(f"Prediction: {result[0]}")

# ================= FUTURE DATASET =================

st.subheader("📁 Batch Prediction (final1.csv)")

if st.button("Run Batch Prediction"):
    dataset1 = pd.read_csv("data/final1.csv")
    d2 = dataset1.copy()

    X_new = dataset1.iloc[:, [3, 4]].values

    # IMPORTANT: use SAME scaler (Normalizer)
    X_new = sc.transform(X_new)

    preds = model.predict(X_new)
    d2['y_pred1'] = preds

    st.write("Predicted Data:")
    st.dataframe(d2)

    d2.to_csv("output/final2.csv", index=False)
    st.success("File saved as output/final2.csv")
