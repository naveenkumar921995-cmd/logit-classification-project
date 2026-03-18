import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("🚀 Logistic Regression Dashboard")
st.markdown("### 📊 Model Analysis + Prediction App")

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

# ================= METRICS =================

st.markdown("## 📈 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Training Accuracy", f"{train_acc:.2f}")
col3.metric("Testing Accuracy", f"{test_acc:.2f}")

# ================= CONFUSION MATRIX =================

st.markdown("## 🔍 Confusion Matrix")

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# ================= SINGLE PREDICTION =================

st.markdown("## 🔮 Single Prediction")

col1, col2 = st.columns(2)

f1 = col1.number_input("Feature 1")
f2 = col2.number_input("Feature 2")

if st.button("Predict"):
    data = np.array([[f1, f2]])
    data = sc.transform(data)
    result = model.predict(data)
    st.success(f"Prediction: {result[0]}")

# ================= FILE UPLOAD =================

st.markdown("## 📁 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", data)

    try:
        X_new = data.iloc[:, [3, 4]].values
        X_new = sc.transform(X_new)

        preds = model.predict(X_new)
        data['Prediction'] = preds

        st.write("✅ Prediction Result", data)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Predictions",
            csv,
            "predicted_data.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("⚠️ Ensure your CSV has correct column format")

# ================= FOOTER =================

st.markdown("---")
st.markdown("👨‍💻 Developed by Naveen Kumar")
