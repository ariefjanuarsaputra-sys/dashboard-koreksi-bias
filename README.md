# 🌦️ Dashboard Analisis & Koreksi Bias Curah Hujan

Dashboard interaktif berbasis **Streamlit** untuk menganalisis dan melakukan koreksi bias data curah hujan model iklim **ERA5** terhadap data observasi **BMKG**, dilengkapi analisis indeks ekstrem klimatologi (ETCCDI) dan kurva distribusi CDF.

---

## 📍 Lokasi & Periode Data

| Info | Detail |
|------|--------|
| Stasiun | Meteorologi Minangkabau |
| Kabupaten | Padang Pariaman, Sumatera Barat |
| Koordinat | -0.7866°, 100.2813° |
| Elevasi | 3 m dpl |
| Sumber Model | ERA5 Reanalysis (ECMWF) |
| Periode | 1991 – 2025 |

---

## ✨ Fitur Dashboard

### 🔧 Metode Koreksi Bias (6 Metode)
| Metode | Deskripsi |
|--------|-----------|
| **Sebelum Koreksi (ERA5)** | Data ERA5 mentah tanpa koreksi (baseline) |
| **Delta Method** | Koreksi berbasis selisih rata-rata bulanan dengan faktor bobot yang dapat diatur |
| **Linear Scaling** | Penyesuaian skala linier berdasarkan rasio rata-rata bulanan |
| **Variance Scaling** | Koreksi mean sekaligus variansi distribusi per bulan |
| **Quantile Mapping** | Pemetaan distribusi kuantil model ke distribusi observasi |
| **Detrended Quantile Mapping** | Quantile Mapping dengan detrending tren temporal terlebih dahulu |
| **Quantile Delta Mapping** | Kombinasi delta scaling berbasis kuantil |

### 📊 Visualisasi & Analisis
- **Kartu Metrik Statistik** — MAE, MSE, RMSE, MBE, NSE dengan interpretasi otomatis
- **Grafik Deret Waktu** — perbandingan harian/bulanan dengan filter rentang tanggal
- **Klimatologi Bulanan** — pola rata-rata 3 kondisi (observasi, ERA5 mentah, setelah koreksi)
- **Scatter Plot 1:1** — analisis regresi linier dengan persamaan & R²
- **Indeks Ekstrem ETCCDI** — Rx1day, CDD, CWD per tahun dalam bentuk grafik batang
- **Kurva Distribusi CDF** — perbandingan distribusi kumulatif empiris
- **Mode Komparasi** — perbandingan performa statistik semua metode sekaligus dalam satu tabel
- **Ekspor Data** — download hasil koreksi format `.CSV`

---

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/ariefjanuarsaputra-sys/dashboard-koreksi-bias.git
cd dashboard-koreksi-bias
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Siapkan file data

Pastikan dua file Excel berikut ada di folder yang sama dengan script:

```
dashboard-koreksi-bias/
├── dashboard_koreksi_bias.py
├── Minang_obs 1991-2025.xlsx   ← data observasi BMKG
├── Book1.xlsx                  ← data model ERA5
├── requirements.txt
└── README.md
```

### 4. Jalankan dashboard

```bash
streamlit run dashboard_koreksi_bias.py
```

Buka browser dan akses `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
plotly
scikit-learn
openpyxl
```

---

## 📈 Panduan Interpretasi Metrik

| Metrik | Interpretasi |
|--------|-------------|
| **MAE ↓** | Makin kecil → deviasi model makin dekat ke observasi |
| **RMSE ↓** | Makin kecil → akurasi model makin tinggi |
| **MBE → 0** | Makin mendekati 0 → bias model makin hilang |
| **NSE > 0.70** | ✅ Performa sangat baik |
| **NSE 0.40–0.70** | ⚠️ Performa cukup baik |
| **NSE < 0.40** | ❌ Perlu perbaikan metode |

---

## 📊 Indeks Ekstrem ETCCDI

| Indeks | Nama Lengkap | Satuan |
|--------|-------------|--------|
| **Rx1day** | Max 1-day Precipitation Amount | mm/hari |
| **CDD** | Consecutive Dry Days | Hari |
| **CWD** | Consecutive Wet Days | Hari |

---

## 👤 Author

**Arief Januar Saputra**  
GitHub: [@ariefjanuarsaputra-sys](https://github.com/ariefjanuarsaputra-sys)

---

## 📄 Lisensi

Project ini dibuat untuk keperluan Project Based Learning Mata Kuliah Metode Klimatologi.  
Data observasi bersumber dari **BMKG** dan data model dari **ERA5 Reanalysis (ECMWF)**.
