# Strategi Penurunan Risiko Kredit Macet Berdasarkan Segmentasi Perilaku Finansial Pelanggan

**Tech Stack & Domain:** <code>Python 3.9+</code> | <code>Fintech / Credit Risk</code> | <code>Clustering & Classification (Two-Stage Hybrid Framework)</code>

### Anggota Tim (Data Science IF-A)

| No | Nama | NIM | USN GitHub | Kelas Praktikum |
| :---: | :--- | :---: | :---: | :---:|
| 1 | **Ananda Rizky Setya Nugroho** | `123240070` | Yaboku29 | IF-B |
| 2 | **Fahmi Firdaus** | `123240055` | FahmiFirdaus7 | IF-F |
| 3 | **Nailah Hana Nur'aisyah** | `123240088` | nailahhana | IF-A |
| 4 | **Muhamad Ifsal Fadhillah** | `123240138` | Mesalllnya | IF-F |

---

## Struktur Repositori

```text
.
├── dashboard/
│   ├── dashboard.py                    # Script utama aplikasi Streamlit Dashboard
│   ├── data_dashboard.csv              # Data hasil pemodelan yang otomatis diekspor saat notebook di-run all
│   └── requirements.txt                # Dependensi library untuk dashboard
├── notebook/
│   ├── notebook.ipynb                  # Notebook utama proyek Data Science
│   └── *.png                           # Visualisasi grafik EDA, clustering, dan evaluasi
├── data/
│   └── data.csv                        # Dataset mentah nasabah (50.000 baris, 20 kolom)
└── README.md                           # Dokumentasi komprehensif proyek
```

---

## 1. Business Understanding

### 1.1 Latar Belakang & Masalah
Kredit macet (*Non-Performing Loan* / NPL) merupakan ancaman terbesar bagi stabilitas institusi keuangan perbankan maupun *fintech lending*. Kebijakan persetujuan kredit seragam (*one-size-fits-all*) memiliki kelemahan mendasar: terlalu ketat akan menghilangkan peluang bisnis, sedangkan terlalu longgar akan memicu gagal bayar massal. Proyek ini mengimplementasikan sistem mitigasi risiko cerdas berbasis data (*data-driven*) dengan pendekatan dua tahap:
1. **Segmentasi Perilaku (*Behavioral Segmentation*):** Mengelompokkan nasabah berdasarkan profil demografi dan pola keuangan menggunakan *Unsupervised Learning* (K-Means).
2. **Prediksi Risiko per Segmen (*Segment-Specific Risk Prediction*):** Melatih model klasifikasi *Supervised Learning* (Random Forest) untuk mengestimasi probabilitas gagal bayar (*default*) pada tiap segmen.

### 1.2 Pertanyaan Bisnis (*Business Questions*)
1. Karakteristik finansial dan demografi apa yang paling signifikan membedakan nasabah berisiko gagal bayar dengan nasabah lancar?
2. Bagaimana memetakan dan mengelompokkan nasabah ke dalam segmen perilaku finansial yang homogen, terpisah optimal, dan bermakna secara bisnis?
3. Apakah pemodelan klasifikasi risiko gagal bayar berbasis segmen (*hybrid approach*) memberikan performa deteksi dan fleksibilitas mitigasi yang lebih baik dibandingkan satu model tunggal global?
4. Strategi mitigasi risiko, penyesuaian plafon limit kredit, dan kebijakan suku bunga apa yang tepat diterapkan pada masing-masing segmen nasabah?

### 1.3 Tujuan Bisnis & Metrik Keberhasilan
* **Target Bisnis:** Menurunkan rasio kredit macet (NPL) minimal 15% dari posisi portofolio kredit berjalan.
* **Metrik Teknis Klasifikasi:** Diutamakan pada **Recall (Macet)** untuk menekan *False Negative* (nasabah macet lolos), didukung oleh **Precision**, **F1-Score**, dan **ROC-AUC**.
* **Metrik Teknis Clustering:** **Silhouette Score** dan **Inertia (Elbow Method)**.

---

## 2. Struktur Dataset & Kamus Data

Dataset diperoleh dari Kaggle: [Faisal Surya Saputra (2025) - Dataset Klasifikasi Status Pinjaman](https://www.kaggle.com/datasets/faisalwp/dataset-klasifikasi-status-pinjaman).

| No | Nama Kolom | Tipe Data | Deskripsi & Makna Bisnis |
| :---: | :--- | :---: | :--- |
| 1 | `id_pelanggan` | String | Nomor identifikasi unik nasabah peminjam (*identifier*). |
| 2 | `usia` | Numerik | Usia nasabah saat pengajuan pinjaman (tahun). |
| 3 | `status_pekerjaan` | Kategorikal | Status kepegawaian (Bekerja, Mahasiswa, Wiraswasta, Menganggur, dll). |
| 4 | `lama_bekerja_tahun` | Numerik | Durasi pengalaman kerja nasabah (tahun). |
| 5 | `pendapatan_tahunan` | Numerik | Total penghasilan kotor tahunan nasabah. |
| 6 | `skor_kredit` | Numerik | Nilai kelayakan kredit nasabah (skala FICO 300 - 850). |
| 7 | `lama_riwayat_kredit_tahun` | Numerik | Durasi kepemilikan fasilitas kredit (tahun). |
| 8 | `aset_tabungan` | Numerik | Nilai saldo simpanan likuid atau aset tabungan nasabah. |
| 9 | `hutang_saat_ini` | Numerik | Total saldo kewajiban utang aktif yang sedang berjalan. |
| 10 | `gagal_bayar_tercatat` | Biner | Indikator riwayat pernah gagal bayar sebelumnya (1 = Ya, 0 = Tidak). |
| 11 | `tunggakan_2thn_terakhir` | Numerik | Frekuensi keterlambatan pembayaran dalam 2 tahun terakhir. |
| 12 | `catatan_negatif` | Biner | Indikator catatan reputasi keuangan buruk (1 = Ada, 0 = Tidak). |
| 13 | `tipe_produk` | Kategorikal | Jenis instrumen pembiayaan (Kartu Kredit, Pinjaman Pribadi, Kredit Berjalan). |
| 14 | `tujuan_pinjaman` | Kategorikal | Keperluan penggunaan pinjaman (Bisnis, Pendidikan, Konsolidasi Hutang, dll). |
| 15 | `jumlah_pinjaman` | Numerik | Nominal pokok pinjaman yang diajukan. |
| 16 | `suku_bunga` | Numerik | Tingkat suku bunga tahunan pinjaman (%). |
| 17 | `rasio_hutang_terhadap_pendapatan` | Numerik | *Debt-to-Income Ratio* (DTI): rasio total hutang terhadap pendapatan tahunan. |
| 18 | `rasio_pinjaman_terhadap_pendapatan` | Numerik | *Loan-to-Income Ratio* (LTI): rasio pinjaman yang diajukan terhadap pendapatan. |
| 19 | `rasio_pembayaran_terhadap_pendapatan` | Numerik | *Payment-to-Income Ratio* (PTI): rasio cicilan terhadap pendapatan tahunan. |
| 20 | `status_pinjaman` | Biner | Target: `1 = Lancar` (55.0%), `0 = Macet/Default` (45.0%). Dibalik ke `1 = Macet` pada klasifikasi. |

---

## 3. Data Cleaning, Feature Engineering & Preprocessing

### 3.1 Data Cleaning & Validasi Domain
1. **Imputasi Missing Values:** Imputasi `usia` (147 missing) dengan median usia per kelompok status pekerjaan, dan `status_pekerjaan` (153 missing) dengan modus pekerjaan.
2. **Eliminasi Kolom ID & Whitespace:** Menghapus `id_pelanggan` dan merapikan spasi kategorikal.
3. **Validasi Aturan Domain Perbankan:** Membuang 879 baris yang melanggar logika: 869 baris dengan nilai kolom biner/target bukan 0 atau 1, dan 10 baris dengan lama bekerja melebihi usia produktif. Ukuran bersih: **49.121 baris**.

### 3.2 Feature Engineering (Pembuatan Fitur Baru)
* **`rasio_tabungan_terhadap_pinjaman`:** `aset_tabungan / (jumlah_pinjaman + 1)` mengukur likuiditas penyangga (*safety net/financial cushion*) nasabah.
* **`beban_finansial_total`:** `rasio_hutang_terhadap_pendapatan + rasio_pinjaman_terhadap_pendapatan` menangkap eksposur total *leverage* kewajiban nasabah terhadap kapasitas penghasilannya.

### 3.3 Outlier Capping & Preprocessing
* **Train-Test Split:** Pemisahan 80:20 secara stratified pada target untuk menghindari *data leakage*.
* **Penanganan Outlier (Capping IQR):** Batas kuantil dipelajari HANYA dari data latih, lalu diterapkan pada data latih dan data uji tanpa membuang baris data penting.
* **Transformasi Skala:** `StandardScaler` untuk numerik kontinu, `OneHotEncoder(drop='first')` untuk kategorikal nominal, dan `passthrough` untuk kolom biner.

### 3.4 Feature Selection
Menggunakan konsensus 3 tahap:
1. Penyaringan multikolinearitas ($|r| > 0.85$) yang mengeliminasi `rasio_pinjaman_terhadap_pendapatan` (redundansi dengan rasio cicilan).
2. Voting 3 metode: Mutual Information, Random Forest Importance, dan RFE Logistic Regression.
3. Keputusan akhir: Memilih 14 fitur yang memperoleh minimal 2 suara.

---

## 4. Pemodelan & Hasil Evaluasi

> Backend model dikunci ke Random Forest. Metrik dapat bergeser sekitar 1 poin persentase bila notebook dijalankan dengan versi `scikit-learn` berbeda.

### 4.1 Segmentasi Nasabah (K-Means Clustering)
Berdasarkan *Elbow Method* dan *Silhouette Score* dipilih $K = 3$ segmen. Silhouette K=3 hanya 0.151 (K=2: 0.196), sehingga struktur klaster lemah dan segmen saling tumpang tindih; K=3 dipilih karena kebermaknaan bisnis.

| Segmen | % Nasabah Latih | Tingkat Macet (Latih / Uji) | Usia | Skor Kredit | Suku Bunga | Riwayat Kredit | Aset Tabungan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 1 (Rawan / High Risk)** | 45.7% | **64.8% / 66.1%** | 26.8 | 610.8 | 16.9% | 3.7 thn | 484 |
| **Cluster 0 (Menengah / Standar)** | 31.8% | 26.7% / 24.8% | 42.1 | 683.0 | 13.5% | 10.3 thn | 970 |
| **Cluster 2 (Mapan)** | 22.5% | 26.8% / 26.8% | 41.9 | 660.1 | 15.2% | 14.4 thn | 4.515 |

Cluster 0 dan Cluster 2 memiliki risiko yang setara (sekitar 2.4 kali lebih rendah dari Cluster 1); keduanya berbeda pada profil (Cluster 2 lebih kaya aset dan berpengalaman kredit), bukan pada tingkat risiko.

### 4.2 Hasil Evaluasi Model Klasifikasi Risiko (Data Uji 9.825 Nasabah, threshold 0.50)

| Model / Segmen | n Nasabah Uji | Default Rate Uji | Precision (Macet) | Recall (Macet) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | 3.163 | 0.248 | 0.704 | 0.862 | 0.775 | 0.951 |
| **Cluster 1** | 4.489 | 0.661 | 0.938 | 0.811 | 0.870 | 0.934 |
| **Cluster 2** | 2.173 | 0.268 | 0.793 | 0.889 | 0.838 | 0.971 |
| **Model Hybrid (Gabungan)** | 9.825 | 0.441 | 0.862 | 0.831 | 0.846 | 0.946 |
| Model Tunggal Global | 9.825 | 0.441 | 0.871 | 0.872 | 0.871 | 0.957 |

### 4.3 Optimasi Model (Notebook Bab 8)
Hyperparameter dioptimasi dengan `RandomizedSearchCV` (10 kombinasi, 3-fold CV, skor ROC-AUC) dan threshold dikalibrasi dari prediksi *out-of-fold* data latih dengan **asumsi** biaya False Negative = 3 x False Positive. Data uji hanya dipakai untuk evaluasi akhir.

| Skenario (data uji) | Precision | Recall | F1 | ROC-AUC | Biaya / 1.000 nasabah |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Hybrid awal (thr 0.50) | 0.862 | 0.831 | 0.846 | 0.946 | 282.7 |
| Hybrid tuning (thr 0.50) | 0.868 | 0.849 | 0.858 | 0.952 | 257.1 |
| Hybrid tuning + threshold per cluster (C0 0.45, C1 0.23, C2 0.39) | 0.786 | 0.945 | 0.858 | 0.952 | 186.5 |
| Global awal (thr 0.50) | 0.871 | 0.872 | 0.871 | 0.957 | 227.3 |
| Global tuning (thr 0.50) | 0.872 | 0.884 | 0.878 | 0.962 | 211.3 |
| Global tuning + threshold tunggal (0.33) | 0.801 | 0.943 | 0.866 | 0.962 | 179.2 |

Temuan utama:
* Tuning hyperparameter hanya menaikkan ROC-AUC CV sekitar +0.004 sampai +0.006; kalibrasi threshold berdampak jauh lebih besar.
* Threshold 0.35 pada Cluster 1 menghasilkan Recall 91.4% (bukan > 95%); Recall 96.1% tercapai pada threshold sekitar 0.23 (Precision 83.1%).
* Melonggarkan Cluster 2 ke threshold 0.60 menurunkan Recall menjadi 83.4% dan menaikkan biaya; tidak disarankan.
* Threshold optimal sangat bergantung pada asumsi rasio biaya (lihat tabel sensitivitas di notebook).

### 4.4 Pemilihan Model
Dari sisi akurasi prediksi murni, **Model Global setara atau sedikit lebih baik** daripada Model Hybrid, baik pada konfigurasi awal maupun setelah optimasi. Model Hybrid dipertahankan sebagai kerangka mitigasi karena:
1. **Interpretabilitas:** segmen menjelaskan *mengapa* nasabah berisiko.
2. **Kebijakan terdiferensiasi:** limit, bunga, dan tenor per segmen membutuhkan pengelompokan.
3. **Threshold khusus Cluster 1:** manfaat kalibrasi threshold terpusat pada segmen ini.

Alternatif praktis: Model Global untuk skoring risiko dan K-Means hanya untuk menentukan kebijakan per segmen.

---

## 5. Keputusan Bisnis & Rekomendasi Manajerial

1. **Kebijakan Cluster 1 (High Risk / Rawan):**
   * Batasi limit pinjaman maksimal hingga 30% dari pengajuan standar (maksimal Rp 10 - 15 juta).
   * Turunkan threshold probabilitas macet menjadi sekitar 0.23 (Recall sekitar 96%, asumsi FN = 3 x FP).
   * Terapkan suku bunga berbasis risiko (*risk-based pricing*) dan tenor pendek (< 12 bulan).
   * Wajibkan agunan likuid atau verifikasi penjamin (*guarantor*).
2. **Kebijakan Cluster 0 (Menengah):**
   * Limit proporsional dengan batas rasio cicilan terhadap pendapatan bulanan (PTI < 25%).
   * Notifikasi otomatis H-3 jatuh tempo dan opsi restrukturisasi tenor bila terjadi kendala pembayaran awal.
   * Pertahankan threshold sekitar 0.50.
3. **Kebijakan Cluster 2 (Mapan):**
   * Persetujuan instan hanya untuk nasabah dengan probabilitas macet rendah (mis. < 0.20), bukan seluruh segmen (tingkat macet segmen 26.8%).
   * Jangan melonggarkan threshold ke 0.60.
   * Plafon lebih tinggi dan insentif bunga untuk nasabah berskor probabilitas rendah sebagai strategi retensi.

---

## 6. Cara Menjalankan Proyek

1. **Clone Repositori & Masuk Direktori:**
   ```bash
   git clone <URL_REPOSITORI>
   cd credit-risk-mitigation-segmentation
   ```
2. **Jalankan Jupyter Notebook:**
   ```bash
   jupyter notebook notebook/notebook.ipynb
   ```
   Seluruh tahapan dari eksplorasi, imputasi, pembuatan fitur, clustering, klasifikasi, evaluasi komparatif, konklusi, hingga optimasi model (Bab 8) dapat langsung direproduksi secara utuh. **Ketika notebook ini di-*Run All*, data hasil pemodelan dan prediksi otomatis diekspor langsung ke `dashboard/data_dashboard.csv`**.

3. **Menjalankan Dashboard Interaktif Streamlit:**
   Untuk membuka dan berinteraksi langsung dengan dashboard mitigasi risiko kredit:
   ```bash
   streamlit run dashboard/dashboard.py
   ```
   Dashboard menyediakan fitur interaktif manipulasi data langsung (filter klaster segmen nasabah, slider rentang skor kredit FICO, multiselect kategori tujuan pinjaman, slider rasio DTI, serta simulator ambang batas *decision threshold* yang secara realtime mengubah visualisasi Confusion Matrix dan kurva ROC).
