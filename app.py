"""
=============================================================
 Prediksi Kelulusan Mahasiswa
 Kelompok       : Kelompok 2
 Mata Kuliah    : Machine Learning / Pembelajaran Mesin
 Sumber Dataset : UCI Student Performance Dataset
 Model          : Naive Bayes | Decision Tree | Logistic Regression
 Target         : G3 >= 10 = Lulus | G3 < 10 = Tidak Lulus
 -------------------------------------------------------------
 Diadaptasi dari template:
 Explore-AI/classification-predict-streamlit-template
 https://github.com/Explore-AI/classification-predict-streamlit-template
 Lisensi: MIT License
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* ── Header Banner ── */
    .header-banner {
        background: linear-gradient(135deg, #1a3a5c 0%, #2e6da4 60%, #4a9fd4 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(30,80,140,0.18);
    }
    .header-banner h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .header-banner p {
        margin: 0.4rem 0 0;
        font-size: 1rem;
        opacity: 0.88;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        flex: 1;
        background: white;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        border-left: 5px solid #2e6da4;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1a3a5c;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 2px;
    }

    /* ── Section Title ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a3a5c;
        border-left: 4px solid #2e6da4;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }

    /* ── Result Box ── */
    .result-lulus {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1.3rem 1.8rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #065f46;
        margin-top: 1rem;
    }
    .result-tidak {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1.3rem 1.8rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #991b1b;
        margin-top: 1rem;
    }

    /* ── Info Box ── */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        font-size: 0.88rem;
        color: #1e40af;
        margin-top: 0.7rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS & ENCODERS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    sheet_id = "1-vn6Y3fwi2117wujrlv-zt4YUFvivoCt5gWIL8A-TzQ"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    return df

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🎓 Prediksi Kelulusan Mahasiswa</h1>
    <p>Kelompok 2 &nbsp;·&nbsp; Machine Learning &nbsp;|&nbsp; UCI Student Performance Dataset &nbsp;|&nbsp; Target: Nilai G3 ≥ 10 = Lulus</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Total Dataset</div>
        <div class="value">649</div>
        <div class="sub">Mahasiswa</div>
    </div>
    <div class="metric-card">
        <div class="label">Jumlah Fitur</div>
        <div class="value">33</div>
        <div class="sub">Kolom / Variabel</div>
    </div>
    <div class="metric-card" style="border-left-color:#6366f1;">
        <div class="label">Naive Bayes</div>
        <div class="value">78.46%</div>
        <div class="sub">Akurasi Model</div>
    </div>
    <div class="metric-card" style="border-left-color:#f59e0b;">
        <div class="label">Decision Tree</div>
        <div class="value">86.15%</div>
        <div class="sub">Akurasi Model</div>
    </div>
    <div class="metric-card" style="border-left-color:#10b981;">
        <div class="label">Logistic Regression</div>
        <div class="value">90.00%</div>
        <div class="sub">Akurasi Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Perbandingan Model", "🔮 Prediksi", "📂 Dataset"])

# ══════════════════════════════════════════════
# TAB 1 – MODEL COMPARISON
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Perbandingan Akurasi Model</div>', unsafe_allow_html=True)

    col_chart, col_tbl = st.columns([3, 2], gap="large")

    with col_chart:
        models = ["Naive Bayes", "Decision Tree", "Logistic\nRegression"]
        accuracies = [78.46, 86.15, 90.00]
        colors = ["#6366f1", "#f59e0b", "#10b981"]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")

        bars = ax.barh(models, accuracies, color=colors, height=0.5, edgecolor="white", linewidth=1.5)
        ax.set_xlim(60, 100)
        ax.set_xlabel("Akurasi (%)", fontsize=10, color="#4b5563")
        ax.set_title("Akurasi Model Machine Learning", fontsize=12, fontweight="bold", color="#1a3a5c", pad=12)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.2f}%", va="center", fontsize=10, fontweight="bold", color="#1a3a5c")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, colors="#6b7280")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)

    with col_tbl:
        st.markdown('<div class="section-title">Tabel Akurasi</div>', unsafe_allow_html=True)
        df_acc = pd.DataFrame({
            "Model": ["Naive Bayes", "Decision Tree", "Logistic Regression"],
            "Akurasi": ["78.46%", "86.15%", "90.00%"],
            "Keterangan": ["Baseline", "Lebih Baik", "⭐ Terbaik"],
        })
        st.dataframe(df_acc, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-box">
            <b>Logistic Regression</b> dipilih sebagai model terbaik dengan akurasi
            tertinggi <b>90.00%</b>. Cocok untuk klasifikasi biner seperti Lulus / Tidak Lulus.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 – PREDICTION
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Input Data Mahasiswa</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            school   = st.selectbox("Sekolah", ["GP", "MS"])
            sex      = st.selectbox("Jenis Kelamin", ["F", "M"])
            age      = st.slider("Usia", 15, 22, 17)
            address  = st.selectbox("Tipe Tempat Tinggal", ["U", "R"])
            famsize  = st.selectbox("Ukuran Keluarga", ["LE3", "GT3"])
            Pstatus  = st.selectbox("Status Orang Tua", ["T", "A"])
            Medu     = st.slider("Pendidikan Ibu (0–4)", 0, 4, 2)
            Fedu     = st.slider("Pendidikan Ayah (0–4)", 0, 4, 2)
            Mjob     = st.selectbox("Pekerjaan Ibu", ["teacher","health","services","at_home","other"])
            Fjob     = st.selectbox("Pekerjaan Ayah", ["teacher","health","services","at_home","other"])
            reason   = st.selectbox("Alasan Pilih Sekolah", ["home","reputation","course","other"])

        with col2:
            guardian  = st.selectbox("Wali", ["mother","father","other"])
            traveltime= st.slider("Waktu Perjalanan (1–4)", 1, 4, 1)
            studytime = st.slider("Waktu Belajar (1–4)", 1, 4, 2)
            failures  = st.slider("Gagal Kelas Sebelumnya", 0, 4, 0)
            schoolsup = st.selectbox("Dukungan Sekolah", ["yes","no"])
            famsup    = st.selectbox("Dukungan Keluarga", ["yes","no"])
            paid      = st.selectbox("Kelas Berbayar", ["yes","no"])
            activities= st.selectbox("Kegiatan Ekstrakurikuler", ["yes","no"])
            nursery   = st.selectbox("Pernah TK", ["yes","no"])
            higher    = st.selectbox("Ingin S2/Lanjut", ["yes","no"])
            internet  = st.selectbox("Akses Internet", ["yes","no"])

        with col3:
            romantic  = st.selectbox("Pacaran", ["yes","no"])
            famrel    = st.slider("Hubungan Keluarga (1–5)", 1, 5, 3)
            freetime  = st.slider("Waktu Luang (1–5)", 1, 5, 3)
            goout     = st.slider("Keluar Bersama Teman (1–5)", 1, 5, 3)
            Dalc      = st.slider("Alkohol Hari Kerja (1–5)", 1, 5, 1)
            Walc      = st.slider("Alkohol Akhir Pekan (1–5)", 1, 5, 1)
            health    = st.slider("Kesehatan (1–5)", 1, 5, 3)
            absences  = st.number_input("Jumlah Absen", 0, 100, 0)
            G1        = st.slider("Nilai G1 (0–20)", 0, 20, 10)
            G2        = st.slider("Nilai G2 (0–20)", 0, 20, 10)
            model_choice = st.selectbox(
                "Pilih Model Prediksi",
                ["Naive Bayes", "Decision Tree", "Logistic Regression"],
                index=2,
            )

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

    if submitted:
        # ── Build raw input dict ──────────────────────────────
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

        # ── Encode categorical features ───────────────────────
        input_df = pd.DataFrame([raw])
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        # ── Predict ───────────────────────────────────────────
        model_map = {
            "Naive Bayes": nb_model,
            "Decision Tree": dt_model,
            "Logistic Regression": lr_model,
        }
        chosen_model = model_map[model_choice]
        prediction = chosen_model.predict(input_df)[0]
        proba = chosen_model.predict_proba(input_df)[0] if hasattr(chosen_model, "predict_proba") else None

        # ── Display result ────────────────────────────────────
        st.markdown("---")
        res_col, info_col = st.columns([2, 3])

        with res_col:
            if prediction == 1:
                st.markdown('<div class="result-lulus">✅ LULUS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-tidak">❌ TIDAK LULUS</div>', unsafe_allow_html=True)

        with info_col:
            st.markdown(f"""
            <div class="info-box">
                <b>Model:</b> {model_choice}<br>
                <b>Nilai G1:</b> {G1} &nbsp;|&nbsp; <b>Nilai G2:</b> {G2}<br>
                {"<b>Probabilitas Lulus:</b> " + f"{proba[1]*100:.1f}%" if proba is not None else ""}
                {"&nbsp;|&nbsp; <b>Tidak Lulus:</b> " + f"{proba[0]*100:.1f}%" if proba is not None else ""}
                <br><br>
                <i>Keputusan: G3 ≥ 10 → Lulus | G3 &lt; 10 → Tidak Lulus</i>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 – DATASET PREVIEW
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Preview Dataset</div>', unsafe_allow_html=True)
    try:
        df = load_data()
        st.caption(f"Total data: {len(df)} baris × {len(df.columns)} kolom")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown('<div class="section-title">Statistik Deskriptif</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
    except Exception as e:
        st.warning(f"Gagal memuat dataset dari Google Sheets: {e}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <b>Kelompok 2</b> &nbsp;·&nbsp; Mata Kuliah Machine Learning / Pembelajaran Mesin<br>
    UCI Student Performance Dataset &nbsp;·&nbsp; Naive Bayes | Decision Tree | Logistic Regression<br>
    <span style="font-size:0.75rem;">
        Diadaptasi dari template:
        <a href="https://github.com/Explore-AI/classification-predict-streamlit-template" target="_blank" style="color:#9ca3af;">
            Explore-AI/classification-predict-streamlit-template
        </a>
    </span>
</div>
""", unsafe_allow_html=True)
