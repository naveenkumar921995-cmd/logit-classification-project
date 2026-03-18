import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Model Comparison App", layout="wide")

st.title("🚀 ML Model Comparison Dashboard")
st.markdown("### Compare Scaling Techniques with Logistic Regression")

# ================= LOAD DATA =================
dataset = pd.read_csv("data/logit_classification.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# ================= TRAIN MODELS =================

# 1️⃣ StandardScaler Model
sc1 = StandardScaler()
X_train_sc = sc1.fit_transform(X_train)
X_test_sc = sc1.transform(X_test)

model_sc = LogisticRegression(max_iter=1000)
model_sc.fit(X_train_sc, y_train)
y_pred_sc = model_sc.predict(X_test_sc)

acc_sc = accuracy_score(y_test, y_pred_sc)
cm_sc = confusion_matrix(y_test, y_pred_sc)

# 2️⃣ Normalizer Model
sc2 = Normalizer()
X_train_nm = sc2.fit_transform(X_train)
X_test_nm = sc2.transform(X_test)

model_nm = LogisticRegression(max_iter=1000)
model_nm.fit(X_train_nm, y_train)
y_pred_nm = model_nm.predict(X_test_nm)

acc_nm = accuracy_score(y_test, y_pred_nm)
cm_nm = confusion_matrix(y_test, y_pred_nm)

# 3️⃣ Without Scaling Model
model_ns = LogisticRegression(max_iter=1000)
model_ns.fit(X_train, y_train)
y_pred_ns = model_ns.predict(X_test)

acc_ns = accuracy_score(y_test, y_pred_ns)
cm_ns = confusion_matrix(y_test, y_pred_ns)

# ================= SIDEBAR =================

st.sidebar.title("⚙️ Select Model")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ["StandardScaler", "Normalizer", "No Scaling"]
)

# ================= COMPARISON TABLE =================

st.markdown("## 📊 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["StandardScaler", "Normalizer", "No Scaling"],
    "Accuracy": [acc_sc, acc_nm, acc_ns]
})

st.dataframe(comparison_df)

st.bar_chart(comparison_df.set_index("Model"))

# ================= SELECTED MODEL DETAILS =================

st.markdown("## 🔍 Selected Model Details")

if model_option == "StandardScaler":
    st.subheader("StandardScaler Model")
    st.write(f"Accuracy: {acc_sc:.2f}")
    st.write("Confusion Matrix:")
    st.dataframe(cm_sc)

elif model_option == "Normalizer":
    st.subheader("Normalizer Model")
    st.write(f"Accuracy: {acc_nm:.2f}")
    st.write("Confusion Matrix:")
    st.dataframe(cm_nm)

elif model_option == "No Scaling":
    st.subheader("No Scaling Model")
    st.write(f"Accuracy: {acc_ns:.2f}")
    st.write("Confusion Matrix:")
    st.dataframe(cm_ns)

# ================= BEST MODEL =================

best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]

st.markdown("## 🏆 Best Model")

st.success(f"""
Best Model: {best_model['Model']}  
Accuracy: {best_model['Accuracy']:.2f}
""")

# ================= PREDICTION =================

st.markdown("## 🔮 Make Prediction")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):

    data = np.array([[f1, f2]])

    if model_option == "StandardScaler":
        data = sc1.transform(data)
        result = model_sc.predict(data)

    elif model_option == "Normalizer":
        data = sc2.transform(data)
        result = model_nm.predict(data)

    else:
        result = model_ns.predict(data)

    st.success(f"Prediction: {result[0]}")

# ================= FOOTER =================
st.markdown("---")
st.markdown("👨‍💻 Developed by Naveen Kumar")
