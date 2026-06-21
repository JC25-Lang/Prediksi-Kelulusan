import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.markdown("**Dataset**: UCI Student Performance | **Kelompok 2 Machine Learning**")
st.divider()

@st.cache_resource
def load_assets():
    nb  = joblib.load("model_nb.pkl")
    dt  = joblib.load("model_dt.pkl")
    lr  = joblib.load("model_lr.pkl")
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return nb, dt, lr, encoders

@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1-vn6Y3fwi2117wujrlv-zt4YUFvivoCt5gWIL8A-TzQ/export?format=csv"
    data = pd.read_csv(url)
    return data

nb_model, dt_model, lr_model, encoders = load_assets()

with st.spinner("Memuat dataset..."):
    data = load_data()

# Preprocessing sama persis seperti di Colab
X = data.drop(columns=['G3'])
y = (data['G3'] >= 10).astype(int)

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hitung akurasi
acc_nb = round(accuracy_score(y_test, nb_model.predict(X_test)) * 100, 2)
acc_dt = round(accuracy_score(y_test, dt_model.predict(X_test)) * 100, 2)
acc_lr = round(accuracy_score(y_test, lr_model.predict(X_test)) * 100, 2)

# ── Bar Chart ─────────────────────────────────────────────
st.subheader("📊 Perbandingan Akurasi Model")

chart_data = pd.DataFrame({
    "Akurasi (%)": [acc_nb, acc_dt, acc_lr]
}, index=["Naive Bayes", "Decision Tree", "Logistic Regression"])

st.bar_chart(chart_data, color="#1E90FF")

# ── Tabel ─────────────────────────────────────────────────
st.subheader("📋 Tabel Hasil Model")

hasil = pd.DataFrame({
    "Model": ["Naive Bayes", "Decision Tree", "Logistic Regression"],
    "Accuracy (%)": [acc_nb, acc_dt, acc_lr]
}).sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)

hasil.index += 1
st.dataframe(hasil, use_container_width=True)

# ── Model Terbaik ─────────────────────────────────────────
best = hasil.iloc[0]
st.success(f"✅ Model terbaik: **{best['Model']}** dengan akurasi **{best['Accuracy (%)']}%**")

st.caption("G3 ≥ 10 → Lulus | G3 < 10 → Tidak Lulus")
