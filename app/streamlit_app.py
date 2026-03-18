import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ================= CONFIG =================
st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("🚀 Logistic Regression Dashboard")
st.markdown("### 📊 Model Analysis + Prediction")

MODEL_PATH = "models/logistic_model.pkl"

# ================= LOAD OR TRAIN MODEL =================
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model, sc = pickle.load(f)
            return model, sc
        except:
            st.warning("⚠️ Corrupted model found. Retraining...")

    # Train new model
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

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, sc), f)

    return model, sc


model, sc = load_or_train_model()

# ================= MODEL METRICS =================
dataset = pd.read_csv("data/logit_classification.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

X_scaled = sc.transform(X)
y_pred = model.predict(X_scaled)

cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

# ================= METRICS UI =================
st.markdown("## 📈 Model Performance")

col1, col2 = st.columns(2)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Total Records", len(y))

# ================= CONFUSION MATRIX =================
st.markdown("## 🔍 Confusion Matrix")

cm_df = pd.DataFrame(cm)
st.dataframe(cm_df)
st.bar_chart(cm_df)

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

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

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
