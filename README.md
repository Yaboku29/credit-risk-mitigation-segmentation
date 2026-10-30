# Strategi Penurunan Risiko Kredit Macet Berdasarkan Segmentasi Perilaku Finansial Pelanggan

**Tech Stack & Domain:** <code>Python 3.9+</code> | <code>Fintech / Credit Risk</code> | <code>Clustering & Classification</code>

### Anggota Tim (Data Science IF-A)

| No | Nama | NIM | USN GitHub |Kelas Praktikum|
| :---: | :--- | :---: | :---: |:---:|
| 1 | **Ananda Rizky Setya Nugroho** | `123240070` | Yaboku29|IF-B|
| 2 | **Fahmi Firdaus** | `123240055` |FahmiFirdaus7|IF-F|
| 3 | **Nailah Hana Nur'aisyah** | `123240088` |nailahhana|IF-A|
| 4 | **Muhamad Ifsal Fadhillah** | `123240138` | Mesalllnya|IF-F|

---

## Struktur Repositori

```text
.
├── data/               # Folder penyimpanan dataset
|    ├── data.csv
├── README.md           # Dokumentasi proyek
└── test.ipynb          # Notebook eksplorasi data, eksperimen model, & segmentasi
```

---

## 1. Latar Belakang & Tujuan
Kredit macet (*Non-Performing Loan*) merupakan salah satu risiko terbesar dalam industri keuangan dan *fintech lending*. Menggunakan pendekatan satu kebijakan untuk semua pelanggan (*one-size-fits-all*) sering kali tidak efektif dan bisa mengurangi potensi profit bisnis.

Proyek ini bertujuan untuk **memitigasi risiko kredit macet** dengan pendekatan dua tahap:
1. **Segmentasi Perilaku:** Mengelompokkan nasabah berdasarkan profil demografi dan kebiasaan finansial mereka menggunakan algoritma *Unsupervised Learning*.
2. **Prediksi Risiko:** Membangun model klasifikasi untuk memprediksi probabilitas gagal bayar (*default*) pada masing-masing segmen nasabah.

Dengan hasil ini, tim manajemen risiko dapat menerapkan strategi intervensi dan kebijakan limit kredit yang disesuaikan secara spesifik (*personalized risk mitigation*).

---

## 2. Struktur Dataset & Fitur
Dataset yang digunakan mencakup informasi detail mengenai profil keuangan dan demografi pelanggan yang mengajukan pinjaman:

* **Fitur Demografi:** Usia, pendapatan bulanan, status pernikahan, status kepemilikan rumah, dan jenis pekerjaan.
* **Fitur Finansial:** Rasio utang terhadap pendapatan (*Debt-to-Income Ratio*), total pinjaman aktif, nilai tabungan, dan pemanfaatan kartu kredit (*credit utilization rate*).
* **Riwayat Kredit:** Jumlah keterlambatan pembayaran sebelumnya, durasi riwayat kredit, dan skor kredit historis.
* **Variabel Target:** `loan_status` (0 = Lancar, 1 = Gagal Bayar/Macet).

---

## 3. Metodologi & Alur Kerja
Proyek ini diselesaikan dengan alur kerja *end-to-end* sebagai berikut:

1. **Exploratory Data Analysis (EDA):** Analisis distribusi data, penanganan data hilang (*missing values*), analisis korelasi, dan deteksi *outliers*.
2. **Feature Engineering:** Normalisasi skala data numerik, *encoding* variabel kategorikal, dan ekstraksi rasio finansial krusial.
3. **Customer Segmentation (Clustering):** Penerapan algoritma **K-Means / K-Medoids** dibantu oleh *Elbow Method* dan *Silhouette Score* untuk menentukan jumlah segmen perilaku keuangan yang optimal.
4. **Risk Predictive Modeling (Classification):** Melatih algoritma klasifikasi seperti **XGBoost / Random Forest** pada tiap segmen untuk memprediksi risiko gagal bayar dengan fokus metrik optimasi pada **Recall** dan **ROC-AUC**.

---

## 4. Cara Menjalankan Proyek

1. **Clone Repositori & Persiapan Environment:**
   ```bash
   git clone <URL_REPOSITORI>
   cd <NAMA_FOLDER>
   ```

2. **Pengaturan Environment File:**
   Pastikan file `.env` sudah dikonfigurasi jika terdapat kredensial atau parameter jalur data tertentu.

3. **Menjalankan Jupyter Notebook:**
   Buka dan jalankan alur analisis serta eksperimen pada file notebook:
   ```bash
   jupyter notebook test.ipynb
   ```
