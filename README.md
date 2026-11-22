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
│   └── notebook.ipynb                  # Notebook utama proyek Data Science
├── data/
│   └── data.csv                        # Dataset mentah nasabah (50.000 baris, 20 kolom)
├── README.md                           # Dokumentasi komprehensif proyek
└── *.png                               # Visualisasi grafik EDA, clustering, dan evaluasi
```

---

## 1. Business Understanding

### 1.1 Latar Belakang & Masalah
Kredit macet (*Non-Performing Loan* / NPL) merupakan ancaman terbesar bagi stabilitas institusi keuangan perbankan maupun *fintech lending*. Kebijakan persetujuan kredit seragam (*one-size-fits-all*) memiliki kelemahan mendasar: terlalu ketat akan menghilangkan peluang bisnis, sedangkan terlalu longgar akan memicu gagal bayar massal. Proyek ini mengimplementasikan sistem mitigasi risiko cerdas berbasis data (*data-driven*) dengan pendekatan dua tahap:
1. **Segmentasi Perilaku (*Behavioral Segmentation*):** Mengelompokkan nasabah berdasarkan profil demografi dan pola keuangan menggunakan *Unsupervised Learning* (K-Means).
2. **Prediksi Risiko per Segmen (*Segment-Specific Risk Prediction*):** Melatih model klasifikasi *Supervised Learning* (Random Forest / XGBoost) untuk mengestimasi probabilitas gagal bayar (*default*) pada tiap segmen.

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
3. **Validasi Aturan Domain Perbankan:** Membuang 879 baris yang melanggar logika (nilai negatif, usia tidak wajar, lama bekerja melebihi usia produktif). Ukuran bersih: **49.121 baris**.

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
3. Keputusan akhir: Memilih 15 fitur yang memperoleh minimal 2 suara.

---

## 4. Pemodelan & Hasil Evaluasi

### 4.1 Segmentasi Nasabah (K-Means Clustering)
Berdasarkan analisis *Elbow Method* dan *Silhouette Score*, dipilih $K = 3$ segmen:
* **Cluster 1 (Segmen Rawan / High Risk - 35.1%):** Usia muda (27.9 thn), skor kredit rendah (589.6), suku bunga tinggi (18.4%), riwayat tunggakan tinggi. **Tingkat Macet = 66.1%**.
* **Cluster 0 (Segmen Menengah / Standar - 34.7%):** Usia dewasa muda (33.0 thn), skor kredit baik (676.5), didominasi pinjaman pribadi. **Tingkat Macet = 24.8%**.
* **Cluster 2 (Segmen Mapan - 30.2%):** Usia matang (45.7 thn), masa kerja & riwayat kredit panjang (> 14 thn), pendapatan & tabungan tertinggi. **Tingkat Macet = 26.8%**.

### 4.2 Hasil Evaluasi Model Klasifikasi Risiko (Data Uji 9.825 Nasabah)

| Model / Segmen | Karakteristik Segmen | n Nasabah Uji | Default Rate Uji | Precision (Macet) | Recall (Macet) | F1-Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | Segmen Menengah (Standar) | 3.163 | 0.248 | 0.711 | 0.856 | 0.777 | 0.950 |
| **Cluster 1** | Segmen Rawan (High Risk) | 4.489 | 0.661 | 0.933 | 0.818 | 0.872 | 0.934 |
| **Cluster 2** | Segmen Mapan (Low Risk) | 2.173 | 0.268 | 0.796 | 0.882 | 0.836 | 0.969 |
| **Model Hybrid (Gabungan)** | **Arsitektur Dua Tahap Berbasis Segmen** | **9.825** | **0.441** | **0.862** | **0.833** | **0.848** | **0.947** |
| Model Tunggal Global | Baseline Pembanding tanpa Segmentasi | 9.825 | 0.441 | 0.871 | 0.870 | 0.870 | 0.957 |

### 4.3 Pemilihan Model Terbaik
**Model Hybrid Berbasis Segmen dipilih sebagai solusi terbaik untuk implementasi perbankan/fintech** karena:
1. **Fleksibilitas Ambang Batas (*Tailored Thresholding*):** Memungkinkan penerapan threshold ketat (0.35) pada Cluster 1 untuk menjaring > 95% nasabah macet, dan threshold longgar (0.60) pada Cluster 2 guna mempermudah nasabah prima.
2. **Intervensi Bisnis Nyata (*Actionability*):** Model memberikan pemahaman konteks *mengapa* nasabah berisiko berdasarkan kelompok perilakunya, bukan sekadar skor probabilitas tunggal.

---

## 5. Keputusan Bisnis & Rekomendasi Manajerial

1. **Kebijakan Cluster 1 (High Risk / Rawan):**
   * Batasi limit pinjaman maksimal hingga maksimal 30% dari pengajuan standar (maksimal Rp 10 - 15 juta).
   * Terapkan suku bunga berbasis risiko yang lebih tinggi (*risk-based pricing*) dan tenor pinjaman pendek (< 12 bulan).
   * Wajibkan agunan likuid atau verifikasi penjamin (*guarantor*).
2. **Kebijakan Cluster 0 (Medium Risk / Menengah):**
   * Limit proporsional dengan batas rasio cicilan terhadap pendapatan bulanan (PTI < 25%).
   * Notifikasi otomatis H-3 jatuh tempo dan opsi restrukturisasi tenor bila terjadi kendala pembayaran awal.
3. **Kebijakan Cluster 2 (Low Risk / Mapan):**
   * Jalur persetujuan instan (*instant pre-approval*) tanpa proses birokrasi berulang.
   * Plafon pinjaman maksimal dan insentif diskon bunga kompetitif untuk retensi nasabah bernilai tinggi.

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
   Seluruh tahapan dari eksplorasi, imputasi, pembuatan fitur, clustering, klasifikasi, evaluasi komparatif, hingga konklusi dapat langsung direproduksi secara utuh. **Ketika notebook ini di-*Run All*, data hasil pemodelan dan prediksi otomatis diekspor langsung ke `dashboard/data_dashboard.csv`**.

3. **Menjalankan Dashboard Interaktif Streamlit:**
   Untuk membuka dan berinteraksi langsung dengan dashboard mitigasi risiko kredit:
   ```bash
   streamlit run dashboard/dashboard.py
   ```
   Dashboard menyediakan fitur interaktif manipulasi data langsung (filter klaster segmen nasabah, slider rentang skor kredit FICO, multiselect kategori tujuan pinjaman, slider rasio DTI, serta simulator ambang batas *decision threshold* yang secara realtime mengubah visualisasi Confusion Matrix dan kurva ROC).
