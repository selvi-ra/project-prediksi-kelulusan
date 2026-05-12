import streamlit as st
import pandas as pd
import joblib
import os

# 1. SETTING HALAMAN
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="wide")

# 2. LOAD MODEL
# Menyesuaikan path karena app.py ada di root, model ada di folder models
model_path = 'models/model_prediksi_kelulusan.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model tidak ditemukan! Pastikan file .pkl ada di folder models.")

#  3. INTERFACE APLIKASI
st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    tahun_masuk = st.number_input("Tahun Masuk", min_value=2015, max_value=2026, value=2022)
    ipk = st.number_input("IPK Terakhir", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    sks = st.number_input("Total SKS Lulus", min_value=0, max_value=160, value=100)
    ips = st.number_input("IPS Semester Terakhir", min_value=0.0, max_value=4.0, value=3.0, step=0.01)

with col2:
    
    # Buat kamus keterangan
    opsi_skripsi = {
    "0: Belum Ambil / Judul Belum ACC": 0,
    "1: Judul ACC / Menyusun Proposal": 1,
    "2: Seminar Proposal (Sempro)": 2,
    "3: Penelitian / Seminar Hasil (Semhas)": 3,
    "4: Sidang Akhir / Yudisium": 4
}

# Tampilkan pilihan teks, tapi simpan nilainya sebagai angka
    status_label = st.selectbox("Progress Skripsi", options=list(opsi_skripsi.keys()))
    status_skripsi = opsi_skripsi[status_label]
    jml_bimbingan = st.number_input("Jumlah Bimbingan", min_value=0, max_value=100, value=0)
    mk_gagal = st.number_input("Jumlah MK Gagal", min_value=0, max_value=50, value=0)
    status_kerja = st.selectbox("Status Kerja", ["Tidak", "Ya"])
    
    status_kerja_val = 1 if status_kerja == "Ya" else 0

# 4. PROSES FEATURE ENGINEERING (Harus LENGKAP sesuai saat Fit Model)
progress_sks = sks / 144
trend_ip = 1 if ips >= ipk else 0
rata_ip = (ipk + ips) / 2

# Menyiapkan data untuk prediksi (URUTAN DAN NAMA KOLOM HARUS SAMA DENGAN X_train)
input_data = pd.DataFrame([{
    'Tahun_Masuk': tahun_masuk,
    'IPK': ipk,
    'IPS_Terakhir': ips,
    'MK_Gagal': mk_gagal,
    'SKS_Lulus': sks,
    'Status_Skripsi': status_skripsi,
    'Jml_Bimbingan': jml_bimbingan,
    'Status_Kerja': status_kerja_val,
    'Progress_SKS': progress_sks,
    'Trend_IP': trend_ip,
    'Rata_IP': rata_ip
}])
# 5. TOMBOL PREDIKSI
if st.button("Cek Prediksi Kelulusan"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] # Peluang tepat waktu
    
    st.divider()
    
    if prediction[0] == 1:
        st.success(f"### Hasil: **TEPAT WAKTU**")
        st.write(f"Tingkat Keyakinan AI: {probability * 100:.2f}%")
    else:
        st.error(f"### Hasil: **TERLAMBAT**")
        st.write(f"Tingkat Keyakinan AI: {(1 - probability) * 100:.2f}%")

st.info("Catatan: Prediksi ini berdasarkan pola data historis mahasiswa sebelumnya.")