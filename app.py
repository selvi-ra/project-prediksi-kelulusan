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
st.write("Silakan masukkan data akademik mahasiswa di bawah ini untuk melihat prediksi kelulusan dan rekomendasi strategi.")

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

st.markdown("<br>", unsafe_allow_html=True)

# 5. MEMBUAT TOMBOL DI TENGAH HALAMAN
# Menggunakan 3 kolom kosong, kolom tengah diisi tombol agar posisinya center
col_b1, col_b2, col_b3 = st.columns([1.5, 1, 1.5])

with col_b2:
    tombol_prediksi = st.button("🚀 Cek Prediksi Kelulusan", type="primary", use_container_width=True)

# 6. PROSES TOMBOL & OUTPUT EARLY WARNING SYSTEM
if tombol_prediksi:
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] # Peluang tepat waktu
    
    st.divider()
    st.subheader("📊 Hasil Analisis & Sistem Peringatan Dini")
    
    if prediction[0] == 1:
        # TAMPILAN JIKA TEPAT WAKTU
        st.success("### 🎉 Status: **Kategori Aman (Berpotensi Tepat Waktu)**")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Peluang Lulus Tepat Waktu", value=f"{probability * 100:.2f}%")
        with col_m2:
            st.write("Grafik Keyakinan Model:")
            st.progress(probability)
            
        st.markdown("""
        **💡 Rekomendasi Pertahankan Performa:**
        * **Konsistensi Nilai:** Pertahankan tren nilai IPS di semester berikutnya agar tidak mengganggu IPK kumulatif.
        * **Manajemen Skripsi:** Lanjutkan bimbingan secara berkala (minimal 1-2 kali seminggu) agar target kelulusan tidak bergeser.
        * **Pengecekan SKS:** Pastikan sisa SKS wajib diambil seluruhnya sesuai paket semester Anda.
        """)
        
    else:
        # TAMPILAN JIKA TERLAMBAT (EARLY WARNING SYSTEM)
        st.error("### ⚠️ PERINGATAN DINI: **Berisiko Terlambat Lulus**")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Tingkat Risiko Keterlambatan", value=f"{(1 - probability) * 100:.2f}%", delta="Perlu Perhatian", delta_color="inverse")
        with col_m2:
            st.write("Skala Risiko Kelulusan:")
            st.progress(1.0 - probability)
            
        # Blok Rekomendasi Otomatis Berdasarkan Input Mahasiswa
        st.warning("#### 📋 Langkah Evaluasi & Strategi Akademik yang Disarankan:")
        
        # Evaluasi 1: Masalah Skripsi atau Bimbingan rendah
        if status_skripsi < 2 or jml_bimbingan < 4:
            st.markdown("- ✍️ **Akselerasi Skripsi:** Progress skripsi atau intensitas bimbingan Anda tergolong rendah. Segera temui atau hubungi dosen pembimbing untuk penentuan judul/revisi proposal agar bisa mengejar jadwal Sempro.")
            
        # Evaluasi 2: Memiliki Mata Kuliah Gagal
        if mk_gagal > 0:
            st.markdown(f"- 📚 **Perbaikan Nilai Kuliah:** Terdapat **{mk_gagal} mata kuliah yang tidak lulus**. Utamakan mengambil kelas remedial atau mengulang mata kuliah tersebut di Semester Pendek atau semester reguler terdekat guna memenuhi syarat kelulusan.")
            
        # Evaluasi 3: Terkendala SKS Rendah dan Status Kerja
        if progress_sks < 0.70 and status_kerja_val == 1:
            st.markdown("- ⏳ **Manajemen Waktu Kuliah & Kerja:** Karena Anda sedang bekerja dan persentase capaian SKS masih di bawah 70%, disarankan menyusun *timeline* harian yang ketat atau berkonsultasi dengan Dosen Wali mengenai strategi pembagian fokus.")
        elif progress_sks < 0.70:
            st.markdown("- 📈 **Akselerasi Pengambilan SKS:** Hubungi program studi untuk memastikan Anda mengambil beban SKS maksimal (maksimal 24 SKS jika IPK mencukupi) guna mengejar ketertinggalan total SKS.")

    st.caption("ℹ️ *Catatan: Prediksi dan rekomendasi ini dihasilkan oleh AI berdasarkan pola data historis mahasiswa sebelumnya. Gunakan hasil ini sebagai bahan evaluasi diri secara bijak.*")