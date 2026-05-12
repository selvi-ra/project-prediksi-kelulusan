import pandas as pd
import numpy as np

n_samples = 500
np.random.seed(42)

# --- 1. FITUR AKADEMIK (HISTORY) ---
# Tahun Masuk (Untuk identitas struktur)
thn_masuk = np.random.choice([2020, 2021, 2022], n_samples)

# IPK & IPS (Kriteria x)
ipk = np.round(np.random.uniform(2.5, 4.0, n_samples), 2)
# IPS kita buat sedikit variatif dari IPK untuk melihat tren nanti
ips_terakhir = np.round(ipk + np.random.uniform(-0.3, 0.3, n_samples), 2)
ips_terakhir = np.clip(ips_terakhir, 0.0, 4.0) # Pastikan tidak lebih dari 4.0

# Jml MK Gagal (Kriteria: <= 2 untuk ontime)
mk_gagal = np.array([np.random.randint(0, 3) if x > 3.0 else np.random.randint(2, 6) for x in ipk])

# SKS Lulus (Kriteria x)
sks_lulus = np.array([np.random.randint(130, 145) if x > 3.2 else np.random.randint(90, 129) for x in ipk])

# Status Skripsi (Skala 0-4 sesuai request: 0:Blm, 1:Judul, 2:Sempro, 3:Semhas, 4:Sidang)
status_skripsi = []
for sks in sks_lulus:
    if sks > 140: status_skripsi.append(4)
    elif sks > 130: status_skripsi.append(3)
    elif sks > 120: status_skripsi.append(2)
    elif sks > 105: status_skripsi.append(1)
    else: status_skripsi.append(0)

# Jumlah Bimbingan (Kriteria: > 10 untuk ontime)
jml_bimbingan = np.array([np.random.randint(11, 20) if s >= 3 else np.random.randint(0, 11) for s in status_skripsi])

# --- 2. FAKTOR EKSTERNAL ---
status_kerja = np.random.choice(['Ya', 'Tidak'], n_samples, p=[0.3, 0.7])

# --- 3. LABEL: TEPAT WAKTU (Target y: "Ya" atau "Tidak") ---
tepat_waktu = []
for i in range(n_samples):
    # MENGIKUTI RULE-BASED DOSEN:
    # Ontime jika: IPK >= 3.0, MK Gagal <= 2, Bimbingan > 10, Skripsi progres tinggi
    if (ipk[i] >= 3.0 and mk_gagal[i] <= 2 and jml_bimbingan[i] > 10 and 
        status_skripsi[i] >= 3 and status_kerja[i] == 'Tidak'):
        tepat_waktu.append('Ya')
    elif (ipk[i] < 2.8 or mk_gagal[i] > 3 or status_skripsi[i] < 2):
        tepat_waktu.append('Tidak')
    else:
        # Area probabilitas (faktor X lainnya)
        tepat_waktu.append(np.random.choice(['Ya', 'Tidak'], p=[0.4, 0.6]))

# --- 4. GABUNGKAN ---
df = pd.DataFrame({
    'NIM': range(120001, 120001 + n_samples),
    'Tahun_Masuk': thn_masuk,
    'IPK': ipk,
    'IPS_Terakhir': ips_terakhir,
    'MK_Gagal': mk_gagal,
    'SKS_Lulus': sks_lulus,
    'Status_Skripsi': status_skripsi,
    'Jml_Bimbingan': jml_bimbingan,
    'Status_Kerja': status_kerja,
    'Tepat_Waktu': tepat_waktu
})

# Simpan ke folder data
df.to_csv('data/dataset_kelulusan_raw.csv', index=False)
print("Dataset sesuai kriteria dosen & faktor eksternal berhasil dibuat!")