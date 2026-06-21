import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600; color: #374151;
    }
    .result-lulus {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-left: 5px solid #10b981;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #065f46;
    }
    .result-tidak {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 5px solid #ef4444;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #7f1d1d;
    }
    .section-header {
        background: #1e3a5f;
        color: white;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & encoders ────────────────────────────────────
@st.cache_resource
def load_assets():
    models = {
        "Naive Bayes": joblib.load("model_nb.pkl"),
        "Decision Tree": joblib.load("model_dt.pkl"),
        "Logistic Regression": joblib.load("model_lr.pkl"),
    }
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return models, encoders

models, encoders = load_assets()

# ── Header ───────────────────────────────────────────────────
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.markdown("Masukkan data mahasiswa di bawah untuk memprediksi apakah akan **Lulus** atau **Tidak Lulus** (berdasarkan nilai G3 ≥ 10).")
st.divider()

# ── Pilih Model ──────────────────────────────────────────────
model_choice = st.selectbox("🤖 Pilih Model Prediksi", list(models.keys()))

st.divider()

# ── Input Form ───────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Data Pribadi & Keluarga</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    school  = st.selectbox("Sekolah", ["GP", "MS"])
    sex     = st.selectbox("Jenis Kelamin", ["F", "M"])
    age     = st.number_input("Usia", min_value=15, max_value=22, value=17)
    address = st.selectbox("Alamat", ["U (Kota)", "R (Desa)"])
    famsize = st.selectbox("Ukuran Keluarga", ["GT3 (>3 orang)", "LE3 (≤3 orang)"])
    Pstatus = st.selectbox("Status Orang Tua", ["T (Bersama)", "A (Terpisah)"])
with col2:
    Medu = st.slider("Pendidikan Ibu (0-4)", 0, 4, 2)
    Fedu = st.slider("Pendidikan Ayah (0-4)", 0, 4, 2)
    Mjob = st.selectbox("Pekerjaan Ibu", ["teacher", "health", "services", "at_home", "other"])
    Fjob = st.selectbox("Pekerjaan Ayah", ["teacher", "health", "services", "at_home", "other"])
    reason   = st.selectbox("Alasan Pilih Sekolah", ["home", "reputation", "course", "other"])
    guardian = st.selectbox("Wali", ["mother", "father", "other"])

st.markdown('<div class="section-header">📚 Data Akademik & Kebiasaan</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    traveltime = st.slider("Waktu Perjalanan ke Sekolah (1-4)", 1, 4, 1)
    studytime  = st.slider("Jam Belajar per Minggu (1-4)", 1, 4, 2)
    failures   = st.slider("Jumlah Kegagalan Kelas Sebelumnya", 0, 3, 0)
    schoolsup  = st.selectbox("Dukungan Ekstra dari Sekolah", ["yes", "no"])
    famsup     = st.selectbox("Dukungan Belajar dari Keluarga", ["yes", "no"])
    paid       = st.selectbox("Ikut Les Berbayar", ["yes", "no"])
    activities = st.selectbox("Ikut Kegiatan Ekstra", ["yes", "no"])
with col4:
    nursery  = st.selectbox("Pernah TK", ["yes", "no"])
    higher   = st.selectbox("Ingin Lanjut ke Perguruan Tinggi", ["yes", "no"])
    internet = st.selectbox("Ada Akses Internet di Rumah", ["yes", "no"])
    romantic = st.selectbox("Sedang Pacaran", ["yes", "no"])
    famrel   = st.slider("Kualitas Hubungan Keluarga (1-5)", 1, 5, 4)
    freetime = st.slider("Waktu Luang Setelah Sekolah (1-5)", 1, 5, 3)
    goout    = st.slider("Sering Keluar dengan Teman (1-5)", 1, 5, 3)

st.markdown('<div class="section-header">🍻 Kesehatan & Nilai</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    Dalc     = st.slider("Konsumsi Alkohol Hari Kerja (1-5)", 1, 5, 1)
    Walc     = st.slider("Konsumsi Alkohol Akhir Pekan (1-5)", 1, 5, 1)
    health   = st.slider("Kondisi Kesehatan (1-5)", 1, 5, 3)
    absences = st.number_input("Jumlah Absensi", min_value=0, max_value=32, value=0)
with col6:
    G1 = st.number_input("Nilai Periode 1 (G1)", min_value=0, max_value=20, value=11)
    G2 = st.number_input("Nilai Periode 2 (G2)", min_value=0, max_value=20, value=11)

st.divider()

# ── Prediksi ─────────────────────────────────────────────────
if st.button("🔍 Prediksi Kelulusan", use_container_width=True, type="primary"):

    # Map display values ke raw values
    raw = {
        "school":     school,
        "sex":        sex,
        "age":        age,
        "address":    address.split(" ")[0],
        "famsize":    famsize.split(" ")[0],
        "Pstatus":    Pstatus.split(" ")[0],
        "Medu":       Medu,
        "Fedu":       Fedu,
        "Mjob":       Mjob,
        "Fjob":       Fjob,
        "reason":     reason,
        "guardian":   guardian,
        "traveltime": traveltime,
        "studytime":  studytime,
        "failures":   failures,
        "schoolsup":  schoolsup,
        "famsup":     famsup,
        "paid":       paid,
        "activities": activities,
        "nursery":    nursery,
        "higher":     higher,
        "internet":   internet,
        "romantic":   romantic,
        "famrel":     famrel,
        "freetime":   freetime,
        "goout":      goout,
        "Dalc":       Dalc,
        "Walc":       Walc,
        "health":     health,
        "absences":   absences,
        "G1":         G1,
        "G2":         G2,
    }

    # Encode kolom kategorikal
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                        'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                        'famsup', 'paid', 'activities', 'nursery', 'higher',
                        'internet', 'romantic']

    for col in categorical_cols:
        raw[col] = encoders[col].transform([raw[col]])[0]

    input_df = pd.DataFrame([raw])
    model = models[model_choice]
    pred  = model.predict(input_df)[0]

    st.markdown("### 📊 Hasil Prediksi")
    if pred == 1:
        st.markdown('<div class="result-lulus">✅ LULUS (G3 ≥ 10)</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('<div class="result-tidak">❌ TIDAK LULUS (G3 < 10)</div>', unsafe_allow_html=True)

    # Tampilkan probabilitas kalau model support
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0]
        st.markdown(f"""
        <br>
        <small>Probabilitas Lulus: <b>{prob[1]*100:.1f}%</b> &nbsp;|&nbsp; Tidak Lulus: <b>{prob[0]*100:.1f}%</b></small>
        """, unsafe_allow_html=True)

st.caption("Dataset: UCI Student Performance | Model: Naive Bayes, Decision Tree, Logistic Regression")
