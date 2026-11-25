# -*- coding: utf-8 -*-
"""
Dashboard Interaktif: Strategi Penurunan Risiko Kredit Macet Berdasarkan Segmentasi Perilaku Finansial Pelanggan
Tech Stack: Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
Lokasi: dashboard/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Credit Risk Mitigation Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mengatur styling responsif yang adaptif untuk Light Mode dan Dark Mode
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. PEMUATAN DATASET DENGAN CACHING
# ==========================================
@st.cache_data
def load_data():
    # Jalur fleksibel: mendeteksi lokasi apakah dijalankan dari folder root atau dari dalam folder dashboard/
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "data_dashboard.csv"),
        os.path.join("dashboard", "data_dashboard.csv"),
        os.path.join("..", "dashboard", "data_dashboard.csv"),
        os.path.join("data", "data_dashboard.csv"),
        os.path.join("..", "data", "data_dashboard.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
            
    # Fallback darurat jika data_dashboard belum ada
    raw_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "data.csv"),
        os.path.join("data", "data.csv"),
        os.path.join("..", "data", "data.csv")
    ]
    for rpath in raw_paths:
        if os.path.exists(rpath):
            df = pd.read_csv(rpath)
            df['usia'] = df['usia'].fillna(df.groupby('status_pekerjaan')['usia'].transform('median')).round().astype(int)
            df['status_pekerjaan'] = df['status_pekerjaan'].fillna(df['status_pekerjaan'].mode()[0])
            if 'id_pelanggan' in df.columns:
                df = df.drop(columns=['id_pelanggan'])
            df['y_risk'] = (df['status_pinjaman'] == 0).astype(int)
            df['cluster'] = 0
            df['cluster_nama'] = 'Cluster 0 (Standar)'
            df['proba_macet'] = 0.5
            df['split'] = 'Data Uji'
            return df
            
    raise FileNotFoundError("File data_dashboard.csv tidak ditemukan di direktori manapun!")

df_master = load_data()


# ==========================================
# 3. SIDEBAR: FITUR INTERAKTIF FILTERING
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.title("🎛️ Filter Eksplorasi Data")
st.sidebar.markdown("Manipulasi parameter di bawah ini untuk melihat **perubahan langsung pada visualisasi grafik**.")

# FITUR INTERAKTIF 1: Pilihan Subset Data (Default: Data Uji hasil terbaik evaluasi model)
pilihan_split = st.sidebar.radio(
    "1. Pilihan Evaluasi Subset Data:",
    options=["Data Uji Saja (Hasil Evaluasi Model)", "Seluruh Data (49.121 Nasabah)", "Data Latih Saja"],
    index=0,
    help="Default 'Data Uji' mencerminkan hasil model terbaik yang dievaluasi pada data unseen."
)

if pilihan_split == "Data Uji Saja (Hasil Evaluasi Model)":
    df_current_subset = df_master[df_master['split'] == 'Data Uji'].copy()
elif pilihan_split == "Data Latih Saja":
    df_current_subset = df_master[df_master['split'] == 'Data Latih'].copy()
else:
    df_current_subset = df_master.copy()

# FITUR INTERAKTIF 2: Filter Segmen Nasabah (Clustering)
daftar_segmen = sorted(df_master['cluster_nama'].dropna().unique().tolist())
segmen_terpilih = st.sidebar.multiselect(
    "2. Filter Segmen Nasabah (Cluster):",
    options=daftar_segmen,
    default=daftar_segmen,
    help="Pilih satu atau beberapa segmen untuk membandingkan profil risiko antarklaster."
)

# FITUR INTERAKTIF 3: Slider Rentang Skor Kredit (FICO)
min_skor = int(df_master['skor_kredit'].min())
max_skor = int(df_master['skor_kredit'].max())
rentang_skor = st.sidebar.slider(
    "3. Rentang Skor Kredit Nasabah (FICO):",
    min_value=min_skor,
    max_value=max_skor,
    value=(min_skor, max_skor),
    step=10,
    help="Geser rentang skor kredit untuk memfilter nasabah. Grafik akan otomatis menyesuaikan."
)

# FITUR INTERAKTIF 4: Filter Kategori Tujuan Pinjaman
daftar_tujuan = sorted(df_master['tujuan_pinjaman'].dropna().unique().tolist())
tujuan_terpilih = st.sidebar.multiselect(
    "4. Kategori Tujuan Pinjaman:",
    options=daftar_tujuan,
    default=daftar_tujuan,
    help="Filter berdasarkan alasan nasabah mengajukan pinjaman."
)

# FITUR INTERAKTIF 5: Filter Tipe Produk Pembiayaan
daftar_produk = sorted(df_master['tipe_produk'].dropna().unique().tolist())
produk_terpilih = st.sidebar.multiselect(
    "5. Tipe Produk Pinjaman:",
    options=daftar_produk,
    default=daftar_produk,
    help="Filter berdasarkan instrumen fasilitas kredit."
)

# FITUR INTERAKTIF 6: Slider Rasio Utang terhadap Pendapatan (DTI)
dti_slider = st.sidebar.slider(
    "6. Maksimal Rasio Utang (DTI):",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="Batasi maksimal Debt-to-Income ratio untuk melihat dampaknya pada profil risiko."
)

# ==========================================
# 4. APLIKASI FILTER KE DATASET
# ==========================================
df_filtered = df_current_subset[
    (df_current_subset['cluster_nama'].isin(segmen_terpilih)) &
    (df_current_subset['skor_kredit'].between(rentang_skor[0], rentang_skor[1])) &
    (df_current_subset['tujuan_pinjaman'].isin(tujuan_terpilih)) &
    (df_current_subset['tipe_produk'].isin(produk_terpilih)) &
    (df_current_subset['rasio_hutang_terhadap_pendapatan'] <= dti_slider)
].copy()

# Pesan peringatan jika filter menghasilkan data kosong
if df_filtered.empty:
    st.error("⚠️ Filter yang Anda pilih menghasilkan 0 data nasabah. Harap longgarkan parameter filter di sidebar.")
    st.stop()


# ==========================================
# 5. HEADER UTAMA & KPI METRICS
# ==========================================
st.title("💳 Strategi Penurunan Risiko Kredit Macet Berdasarkan Segmentasi Perilaku Finansial")
st.markdown("**Dashboard Analisis Interaktif: Segmentasi Nasabah (K-Means) & Prediksi Risiko Gagal Bayar per Segmen (Random Forest)**")

# Menghitung KPI utama berdasarkan data terfilter
total_nasabah = len(df_filtered)
n_macet_riil = int(df_filtered['y_risk'].sum())
n_lancar_riil = total_nasabah - n_macet_riil
default_rate_pct = (n_macet_riil / total_nasabah) * 100 if total_nasabah > 0 else 0
avg_skor_kredit = df_filtered['skor_kredit'].mean()
avg_dti = df_filtered['rasio_hutang_terhadap_pendapatan'].mean()
total_loan_amount = df_filtered['jumlah_pinjaman'].sum()

# Tampilan KPI Ringkas (4 Kolom)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="👥 Total Nasabah Terfilter",
        value=f"{total_nasabah:,} orang",
        delta=f"{(total_nasabah / len(df_current_subset))*100:.1f}% dari subset"
    )

with kpi2:
    st.metric(
        label="⚠️ Tingkat Macet (Default Rate)",
        value=f"{default_rate_pct:.2f}%",
        delta=f"{n_macet_riil:,} nasabah macet",
        delta_color="inverse"
    )

with kpi3:
    st.metric(
        label="📊 Rata-Rata Skor Kredit",
        value=f"{avg_skor_kredit:.1f}",
        delta="Standar FICO 300-850"
    )

with kpi4:
    st.metric(
        label="💰 Total Pokok Pinjaman",
        value=f"Rp {total_loan_amount * 1000:,.0f}",
        delta=f"Rata-rata: Rp {df_filtered['jumlah_pinjaman'].mean() * 1000:,.0f}"
    )

st.markdown("---")


# ==========================================
# 6. BAGIAN 1: VISUALISASI SEBARAN & SEGMENTASI INTERAKTIF
# ==========================================
st.subheader("📈 1. Profiling & Segmentasi Nasabah (Hasil Unsupervised K-Means)")
st.caption("Visualisasi di bawah ini berubah secara dinamis merespons kombinasi filter segmen, skor kredit, dan kategori pinjaman yang dipilih di sidebar.")

col_fig1, col_fig2 = st.columns(2)

with col_fig1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    counts_target = [n_lancar_riil, n_macet_riil]
    labels_target = [f"Lancar (0)\n{n_lancar_riil:,} ({100-default_rate_pct:.1f}%)", 
                     f"Macet (1)\n{n_macet_riil:,} ({default_rate_pct:.1f}%)"]
    colors_pie = ['#2ca02c', '#d62728']
    
    ax1.pie(counts_target, labels=labels_target, colors=colors_pie, autopct='%1.1f%%', 
            startangle=140, explode=(0.04, 0.04), textprops={'fontsize': 10, 'weight': 'bold'})
    ax1.set_title("Proporsi Status Pinjaman pada Data Terfilter", fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)

with col_fig2:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    segmen_summary = df_filtered.groupby('cluster_nama')['y_risk'].agg(['count', 'mean']).reset_index()
    segmen_summary['mean'] = segmen_summary['mean'] * 100
    
    palette_colors = {'Cluster 1 (High Risk / Rawan)': '#d62728', 
                      'Cluster 0 (Menengah / Standar)': '#1f77b4', 
                      'Cluster 2 (Low Risk / Mapan)': '#2ca02c'}
    bar_colors = [palette_colors.get(x, '#6baed6') for x in segmen_summary['cluster_nama']]
    
    bars = ax2.bar(segmen_summary['cluster_nama'], segmen_summary['mean'], color=bar_colors, width=0.55)
    ax2.set_title("Tingkat Gagal Bayar (Default Rate %) per Segmen", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Default Rate (%)")
    ax2.set_ylim(0, max(100, segmen_summary['mean'].max() + 15))
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    st.pyplot(fig2)

col_fig3, col_fig4 = st.columns(2)

with col_fig3:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    tujuan_risk = df_filtered.groupby('tujuan_pinjaman')['y_risk'].mean().sort_values(ascending=False) * 100
    colors_tujuan = ['#d62728' if x > 25 else '#1f77b4' for x in tujuan_risk.values]
    
    ax3.barh(tujuan_risk.index, tujuan_risk.values, color=colors_tujuan, height=0.6)
    ax3.set_title("Risiko Macet (%) Berdasarkan Tujuan Pinjaman Terfilter", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Default Rate (%)")
    ax3.axvline(x=20.0, color='gray', linestyle='--', label='Batas Rata-Rata Acuan (20%)')
    ax3.grid(axis='x', linestyle='--', alpha=0.5)
    ax3.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)

with col_fig4:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='cluster_nama', y='skor_kredit', data=df_filtered, hue='cluster_nama', palette=['#1f77b4', '#d62728', '#2ca02c'], legend=False, ax=ax4)
    ax4.set_title("Distribusi Skor Kredit per Segmen Terfilter", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Segmen")
    ax4.set_ylabel("Skor Kredit (FICO)")
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig4)

st.markdown("---")


# ==========================================
# 7. BAGIAN 2: SIMULATOR KEBIJAKAN & DECISION THRESHOLD
# ==========================================
st.subheader("🎯 2. Simulator Kebijakan Mitigasi Risiko & Ambang Batas (*Decision Threshold*)")
st.markdown("""
Fitur interaktif ini memungkinkan pemangku kepentingan (*Risk Management*) melakukan **simulasi *what-if analysis*** secara langsung.
Geser nilai ambang batas (*threshold*) probabilitas gagal bayar di bawah ini untuk melihat pergeseran **Confusion Matrix, Trade-off Recall vs Precision, dan Estimasi Dampak Finansial**.
""")

sim_col1, sim_col2 = st.columns([1, 2])

with sim_col1:
    st.markdown("##### Pengaturan Parameter Simulasi:")
    threshold_slider = st.slider(
        "Pilih Decision Threshold Klasifikasi Macet:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Default bawaan adalah 0.50. Berdasarkan optimasi Bab 8 (asumsi FN = 3x FP), threshold optimal adalah 0.23 untuk Cluster 1 (High Risk), 0.45 untuk Cluster 0, dan 0.39 untuk Cluster 2."
    )
    
    st.info(f"""
    **Status Threshold Aktif: {threshold_slider:.2f}**
    - Nasabah dengan prediksi probabilitas macet >= {threshold_slider:.2f} diklasifikasikan sebagai **BERISIKO MACET**.
    - Nasabah dengan probabilitas < {threshold_slider:.2f} diputuskan **LANCAR / DISETUJUI**.
    """)
    
    # Biaya kerugian simulasi bisnis
    est_loss_per_default = st.number_input("Estimasi Kerugian per Nasabah Macet (Rp):", value=25000000, step=5000000, format="%d")

# Hitung prediksi kelas berdasarkan threshold interaktif
y_true_sim = df_filtered['y_risk'].values
y_prob_sim = df_filtered['proba_macet'].values
y_pred_sim = (y_prob_sim >= threshold_slider).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_true_sim, y_pred_sim, labels=[0, 1])
tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)

prec, rec, f1, _ = precision_recall_fscore_support(y_true_sim, y_pred_sim, labels=[1], zero_division=0)
rec_val = rec[0] if len(rec) > 0 else 0
prec_val = prec[0] if len(prec) > 0 else 0
f1_val = f1[0] if len(f1) > 0 else 0
auc_val = roc_auc_score(y_true_sim, y_prob_sim) if len(np.unique(y_true_sim)) > 1 else np.nan

with sim_col2:
    # Metrik hasil simulasi pada threshold aktif
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall Macet", f"{rec_val*100:.1f}%", help="Persentase nasabah macet yang berhasil dicegat")
    m2.metric("Precision Macet", f"{prec_val*100:.1f}%", help="Ketepatan penandaan nasabah berisiko")
    m3.metric("F1-Score", f"{f1_val*100:.1f}%")
    m4.metric("ROC-AUC", f"{auc_val:.3f}" if not np.isnan(auc_val) else "N/A")

    # Dua grafik dinamis berdampingan: Confusion Matrix & Kurva ROC Dinamis
    fig_sim, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(10, 3.8))
    
    # Heatmap Confusion Matrix
    cm_display = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_display, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax_cm,
                xticklabels=['Pred Lancar', 'Pred Macet'], yticklabels=['Aktual Lancar', 'Aktual Macet'],
                annot_kws={'fontsize': 11, 'weight': 'bold'})
    ax_cm.set_title(f"Confusion Matrix (Threshold = {threshold_slider:.2f})", fontsize=10, fontweight='bold')
    
    # Kurva ROC dengan titik threshold aktif
    if len(np.unique(y_true_sim)) > 1:
        fpr, tpr, thresholds_roc = roc_curve(y_true_sim, y_prob_sim)
        ax_roc.plot(fpr, tpr, color='#1f77b4', lw=2, label=f"ROC Curve (AUC = {auc_val:.3f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        
        # Titik posisi threshold aktif pada kurva ROC
        current_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        current_tpr = rec_val
        ax_roc.plot(current_fpr, current_tpr, marker='o', markersize=9, color='#d62728', 
                    label=f'Titik Threshold {threshold_slider:.2f}')
        
        ax_roc.set_title("Kurva ROC & Titik Operasi Aktif", fontsize=10, fontweight='bold')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate (Recall)")
        ax_roc.legend(loc='lower right', fontsize=8)
        ax_roc.grid(True, linestyle='--', alpha=0.4)
    else:
        ax_roc.text(0.5, 0.5, "Data hanya berisi satu kelas target", ha='center', va='center')
        
    plt.tight_layout()
    st.pyplot(fig_sim)

# Estimasi Dampak Finansial Bisnis
saved_loss = tp * est_loss_per_default
leaked_loss = fn * est_loss_per_default

st.info(f"""
### 💰 Simulasi Dampak Finansial pada Threshold {threshold_slider:.2f}:
* **Potensi Modal Terselamatkan (True Positive = {tp:,} nasabah):** **Rp {saved_loss:,.0f}** berhasil dicegah dari risiko gagal bayar.
* **Potensi Risiko Kerugian Lolos (False Negative = {fn:,} nasabah):** **Rp {leaked_loss:,.0f}** kredit macet yang lolos pengawasan.
* **Nasabah Lancar Terkena Pengetatan (False Positive = {fp:,} nasabah):** Potensi friksi penolakan/persyaratan ekstra pada nasabah potensial baik.
""")

st.markdown("---")


# ==========================================
# 8. BAGIAN 3: PANDUAN KEBIJAKAN STRATEGIS PER SEGMEN
# ==========================================
st.subheader("📋 3. Panduan Kebijakan Strategis Berdasarkan Segmen Terpilih")

tab1, tab2, tab3 = st.tabs(["🔴 Kebijakan Cluster 1 (High Risk)", "🔵 Kebijakan Cluster 0 (Menengah)", "🟢 Kebijakan Cluster 2 (Low Risk)"])

with tab1:
    st.markdown("""
    **Profil: Segmen Rawan / High Risk (Tingkat Gagal Bayar 64.8% Latih / 66.1% Uji)**
    * **Plafon Kredit Maksimal:** Batasi pinjaman baru maksimal 30% dari pengajuan standar (maksimal Rp 10.000.000 - Rp 15.000.000 atau rasio LTI < 0.20).
    * **Rekomendasi Ambang Batas (*Tailored Threshold*):** Turunkan threshold dari 0.50 menjadi sekitar **0.23** (asumsi FN = 3x FP) untuk mencapai Recall **96.1%** dengan Precision **83.1%** pada data uji (threshold 0.35 hanya mencapai Recall 91.4%).
    * **Suku Bunga & Tenor:** Terapkan *risk-based pricing* dengan suku bunga premi risiko lebih tinggi dan tenor pendek (< 12 bulan).
    * **Persyaratan Wajib:** Wajibkan agunan likuid atau verifikasi penjamin (*guarantor*).
    """)

with tab2:
    st.markdown("""
    **Profil: Segmen Menengah / Peminjam Standar (Tingkat Gagal Bayar 26.7% Latih / 24.8% Uji)**
    * **Limit Pinjaman Proporsional:** Batasi rasio cicilan terhadap pendapatan bulanan (Payment-to-Income / PTI < 25%).
    * **Rekomendasi Ambang Batas:** Pertahankan threshold sekitar **0.50** (threshold validasi 0.45 praktis setara pada data uji).
    * **Sistem Peringatan Dini:** Aktifkan notifikasi otomatis penagihan H-3 sebelum tanggal jatuh tempo angsuran.
    * **Restrukturisasi:** Sediakan fasilitas perpanjangan tenor apabila terjadi indikasi penurunan arus kas pertama kali.
    """)

with tab3:
    st.markdown("""
    **Profil: Segmen Mapan (Tingkat Gagal Bayar 26.8% Latih & Uji — setara Cluster 0, unggul di aset & riwayat kredit)**
    * **Persetujuan Instan Selektif:** Jalur persetujuan cepat hanya diberikan kepada nasabah dengan probabilitas macet rendah menurut model (< 0.20), **bukan untuk seluruh segmen**.
    * **Rekomendasi Ambang Batas:** **Jangan melonggarkan threshold ke 0.60** (menurunkan Recall menjadi 83.4% dan menaikkan biaya kesalahan). Pertahankan threshold di kisaran **0.39 - 0.50**.
    * **Insentif Suku Bunga & Plafon:** Tawarkan plafon lebih tinggi dan bunga kompetitif (8 - 10%) khusus bagi nasabah berskor probabilitas macet rendah untuk strategi retensi.
    * **Penjualan Silang (*Cross-Selling*):** Tawarkan produk premium seperti kartu kredit platinum dan fasilitas pembiayaan investasi bagi nasabah yang lolos kriteria risiko rendah.
    """)

st.caption("Dashboard Proyek Data Science: Strategi Penurunan Risiko Kredit Macet Berdasarkan Segmentasi Perilaku Finansial Pelanggan © 2026")
