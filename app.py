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
import pandas as pd
import pickle

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.markdown("**Dataset**: UCI Student Performance | **Kelompok 2 Machine Learning**")
st.divider()

# ── Akurasi ───────────────────────────────────────────────
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

best = hasil.iloc[0]
st.success(f"✅ Model terbaik: **{best['Model']}** dengan akurasi **{best['Accuracy (%)']}%**")
st.caption("G3 ≥ 10 → Lulus | G3 < 10 → Tidak Lulus")

st.divider()

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("model_nb.pkl", "rb") as f:
        nb = pickle.load(f)
    with open("model_dt.pkl", "rb") as f:
        dt = pickle.load(f)
    with open("model_lr.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        le = pickle.load(f)
    return nb, dt, lr, le

nb_model, dt_model, lr_model, label_encoders = load_models()

# ── Input Prediksi ────────────────────────────────────────
st.subheader("🔮 Prediksi Kelulusan")

col1, col2, col3 = st.columns(3)

with col1:
    school    = st.selectbox("Sekolah", ["GP", "MS"])
    sex       = st.selectbox("Jenis Kelamin", ["F", "M"])
    age       = st.slider("Usia", 15, 22, 17)
    address   = st.selectbox("Tempat Tinggal", ["U", "R"])
    famsize   = st.selectbox("Ukuran Keluarga", ["LE3", "GT3"])
    Pstatus   = st.selectbox("Status Orang Tua", ["T", "A"])
    Medu      = st.slider("Pendidikan Ibu (0-4)", 0, 4, 2)
    Fedu      = st.slider("Pendidikan Ayah (0-4)", 0, 4, 2)
    Mjob      = st.selectbox("Pekerjaan Ibu", ["teacher", "health", "services", "at_home", "other"])
    Fjob      = st.selectbox("Pekerjaan Ayah", ["teacher", "health", "services", "at_home", "other"])
    reason    = st.selectbox("Alasan Pilih Sekolah", ["home", "reputation", "course", "other"])

with col2:
    guardian   = st.selectbox("Wali", ["mother", "father", "other"])
    traveltime = st.slider("Waktu Perjalanan (1-4)", 1, 4, 1)
    studytime  = st.slider("Waktu Belajar (1-4)", 1, 4, 2)
    failures   = st.slider("Pernah Gagal (0-3)", 0, 3, 0)
    schoolsup  = st.selectbox("Dukungan Sekolah", ["yes", "no"])
    famsup     = st.selectbox("Dukungan Keluarga", ["yes", "no"])
    paid       = st.selectbox("Kelas Berbayar", ["yes", "no"])
    activities = st.selectbox("Ekstrakurikuler", ["yes", "no"])
    nursery    = st.selectbox("Pernah TK", ["yes", "no"])
    higher     = st.selectbox("Ingin Lanjut S2", ["yes", "no"])
    internet   = st.selectbox("Akses Internet", ["yes", "no"])

with col3:
    romantic  = st.selectbox("Pacaran", ["yes", "no"])
    famrel    = st.slider("Hub. Keluarga (1-5)", 1, 5, 3)
    freetime  = st.slider("Waktu Luang (1-5)", 1, 5, 3)
    goout     = st.slider("Keluar Teman (1-5)", 1, 5, 3)
    Dalc      = st.slider("Alkohol Kerja (1-5)", 1, 5, 1)
    Walc      = st.slider("Alkohol Akhir Pekan (1-5)", 1, 5, 1)
    health    = st.slider("Kesehatan (1-5)", 1, 5, 3)
    absences  = st.number_input("Jumlah Absen", 0, 32, 0)
    G1        = st.slider("Nilai G1 (0-20)", 0, 20, 10)
    G2        = st.slider("Nilai G2 (0-20)", 0, 20, 10)
    model_choice = st.selectbox(
        "Pilih Model",
        ["Naive Bayes", "Decision Tree", "Logistic Regression"],
        index=2
    )

if st.button("Prediksi Sekarang", use_container_width=True):
    raw = {
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
        "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences, "G1": G1, "G2": G2,
    }

    input_df = pd.DataFrame([raw])

    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    model_map = {
        "Naive Bayes": nb_model,
        "Decision Tree": dt_model,
        "Logistic Regression": lr_model,
    }
    chosen = model_map[model_choice]
    prediction = chosen.predict(input_df)[0]

    if prediction == 1:
        st.success("Prediksi: LULUS")
    else:
        st.error("Prediksi: TIDAK LULUS")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption(
    "Kelompok 2 · Machine Learning · UCI Student Performance Dataset | "
    "Diadaptasi dari Explore-AI/classification-predict-streamlit-template: "
    "https://github.com/Explore-AI/classification-predict-streamlit-template"
)
