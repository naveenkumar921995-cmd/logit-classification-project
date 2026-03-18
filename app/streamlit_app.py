import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="ML Model Comparison App", layout="wide")

st.title("🚀 Advanced ML Model Comparison Dashboard")
st.markdown("### Compare Multiple Models + Upload Your Own Data")

# ================= LOAD DEFAULT DATA =================
dataset = pd.read_csv("data/logit_classification.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# ================= SCALERS =================
scalers = {
    "StandardScaler": StandardScaler(),
    "Normalizer": Normalizer(),
    "No Scaling": None
}

# ================= MODELS =================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

results = []

trained_models = {}

# ================= TRAIN ALL MODELS =================
for scaler_name, scaler in scalers.items():
    for model_name, model in models.items():

        X_train_temp = X_train.copy()
        X_test_temp = X_test.copy()

        if scaler:
            X_train_temp = scaler.fit_transform(X_train_temp)
            X_test_temp = scaler.transform(X_test_temp)

        model.fit(X_train_temp, y_train)
        y_pred = model.predict(X_test_temp)

        acc = accuracy_score(y_test, y_pred)

        results.append({
            "Scaler": scaler_name,
            "Model": model_name,
            "Accuracy": acc
        })

        trained_models[(scaler_name, model_name)] = (model, scaler)

# ================= RESULTS TABLE =================
results_df = pd.DataFrame(results)

st.markdown("## 📊 Model Comparison Table")
st.dataframe(results_df.sort_values(by="Accuracy", ascending=False))

st.markdown("## 📈 Accuracy Comparison Chart")
chart_data = results_df.copy()
chart_data["Model_Config"] = chart_data["Scaler"] + " | " + chart_data["Model"]

st.bar_chart(chart_data.set_index("Model_Config")["Accuracy"])

# ================= BEST MODEL =================
best_row = results_df.loc[results_df['Accuracy'].idxmax()]

st.markdown("## 🏆 Best Model")
st.success(f"""
Scaler: {best_row['Scaler']}  
Model: {best_row['Model']}  
Accuracy: {best_row['Accuracy']:.2f}
""")

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Select Model")

selected_scaler = st.sidebar.selectbox("Select Scaler", list(scalers.keys()))
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

model, scaler = trained_models[(selected_scaler, selected_model)]

# ================= MODEL DETAILS =================
st.markdown("## 🔍 Selected Model Details")

X_test_temp = X_test.copy()
if scaler:
    X_test_temp = scaler.transform(X_test_temp)

y_pred_selected = model.predict(X_test_temp)
cm = confusion_matrix(y_test, y_pred_selected)

st.write(f"Accuracy: {accuracy_score(y_test, y_pred_selected):.2f}")
st.write("Confusion Matrix:")
st.dataframe(cm)

# ================= MANUAL PREDICTION =================
st.markdown("## 🔮 Manual Prediction")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    data = np.array([[f1, f2]])

    if scaler:
        data = scaler.transform(data)

    result = model.predict(data)
    st.success(f"Prediction: {result[0]}")

# ================= FILE UPLOAD =================
st.markdown("## 📁 Upload Your Own Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview", data.head())

    try:
        # Assume last column is target (optional)
        X_new = data.iloc[:, :-1].values

        if scaler:
            X_new = scaler.transform(X_new)

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
        st.error("⚠️ Ensure your dataset has correct feature columns")

# ================= FOOTER =================
st.markdown("---")
st.markdown("👨‍💻 Developed by Naveen Kumar")
