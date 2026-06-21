import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.markdown("**Dataset**: UCI Student Performance | **Kelompok 2 Machine Learning**")
st.divider()

# Akurasi dari hasil training di Colab
acc_nb = 78.46
acc_dt = 86.15
acc_lr = 90.00

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
