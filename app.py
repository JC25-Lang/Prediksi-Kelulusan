# =============================================================
#  Prediksi Kelulusan Mahasiswa
#  Kelompok       : Kelompok 2
#  Mata Kuliah    : Machine Learning / Pembelajaran Mesin
#  Sumber Dataset : UCI Student Performance Dataset
#  Model          : Naive Bayes | Decision Tree | Logistic Regression
#  Target         : G3 >= 10 = Lulus | G3 < 10 = Tidak Lulus
#  -------------------------------------------------------------
#  Diadaptasi dari template:
#  Explore-AI/classification-predict-streamlit-template
#  https://github.com/Explore-AI/classification-predict-streamlit-template
# =============================================================

import streamlit as st
import joblib
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.markdown("Aplikasi prediksi kelulusan mahasiswa berdasarkan nilai menggunakan Machine Learning.")
st.divider()

# ── Akurasi Model (dari Colab) ────────────────────────────
st.subheader("📊 Perbandingan Akurasi Model")

chart_data = pd.DataFrame({
    "Akurasi (%)": [78.46, 86.15, 90.00]
}, index=["Naive Bayes", "Decision Tree", "Logistic Regression"])

st.bar_chart(chart_data, color="#1E90FF")

hasil = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Naive Bayes"],
    "Accuracy (%)": [90.00, 86.15, 78.46]
}).reset_index(drop=True)
hasil.index += 1
st.dataframe(hasil, use_container_width=True)

st.divider()

# ── Prediksi ─────────────────────────────────────────────
st.subheader("🔍 Coba Prediksi Nilai Mahasiswa")

@st.cache_resource
def load_models():
    nb = joblib.load("model_nb.pkl")
    dt = joblib.load("model_dt.pkl")
    lr = joblib.load("model_lr.pkl")
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return nb, dt, lr, encoders

nb_model, dt_model, lr_model, encoders = load_models()

model_pilihan = st.selectbox("Pilih Model", ["Logistic Regression", "Decision Tree", "Naive Bayes"])

col1, col2 = st.columns(2)
with col1:
    g1 = st.number_input("Nilai G1 (Semester 1)", min_value=0, max_value=20, value=10)
with col2:
    g2 = st.number_input("Nilai G2 (Semester 2)", min_value=0, max_value=20, value=10)

if st.button("Prediksi", use_container_width=True, type="primary"):
    # Buat input dengan nilai default untuk fitur lain
    import pandas as pd
    
    # Nilai default (rata-rata/umum)
    input_data = {
        'school': 0, 'sex': 0, 'age': 17, 'address': 1, 'famsize': 0,
        'Pstatus': 0, 'Medu': 2, 'Fedu': 2, 'Mjob': 0, 'Fjob': 0,
        'reason': 0, 'guardian': 0, 'traveltime': 1, 'studytime': 2,
        'failures': 0, 'schoolsup': 0, 'famsup': 1, 'paid': 0,
        'activities': 0, 'nursery': 1, 'higher': 1, 'internet': 1,
        'romantic': 0, 'famrel': 4, 'freetime': 3, 'goout': 3,
        'Dalc': 1, 'Walc': 1, 'health': 3, 'absences': 0,
        'G1': g1, 'G2': g2
    }

    df_input = pd.DataFrame([input_data])

    if model_pilihan == "Logistic Regression":
        model = lr_model
    elif model_pilihan == "Decision Tree":
        model = dt_model
    else:
        model = nb_model

    pred = model.predict(df_input)[0]

    st.markdown("### Hasil Prediksi")
    if pred == 1:
        st.success("✅ **LULUS** — Mahasiswa diprediksi lulus (G3 ≥ 10)")
        st.balloons()
    else:
        st.error("❌ **TIDAK LULUS** — Mahasiswa diprediksi tidak lulus (G3 < 10)")

st.caption("G3 ≥ 10 → Lulus | G3 < 10 → Tidak Lulus | Dataset: UCI Student Performance")
