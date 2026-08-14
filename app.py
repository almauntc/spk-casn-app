from scipy.stats import spearmanr
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from io import BytesIO

# =====================================================
# 🏛️ SISTEM PENDUKUNG KEPUTUSAN FORMASI CASN
# Metode: AHP (Pembobotan) + SAW (Perangkingan)
# =====================================================

# -----------------------------------------------------
# 🔧 Konfigurasi Dasar
st.set_page_config(
    page_title="SPK Formasi CASN - Dashboard",
    page_icon="🏛️",
    layout="wide"
)

# --- INJEKSI CSS UNTUK TAMPILAN PROFESIONAL ---
st.markdown("""
<style>
/* 1. Style Judul Utama */
.title-main {
    color: #004D99; /* Biru Korporat */
    border-bottom: 3px solid #FFC300; /* Garis Kuning sebagai Aksen */
    padding-bottom: 5px;
    margin-bottom: 20px;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
}

/* 2. Style untuk Kotak Metrik */
[data-testid="stMetric"] {
    background-color: #f0f8ff; /* Light Blue BG */
    border: 1px solid #004D99;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# 📚 DEFINISI KRITERIA & FUNGSI PERHITUNGAN SPK
# -----------------------------------------------------

# Definisi Kriteria dan Matriks AHP dari kode Anda
AHP_CRITERIA_NAMES = ['kebutuhan_ideal', 'pensiun_5th', 'eksisting_asn', 'eksisting_non_asn', 
                      'rasio_kebutuhan_unor', 'nilai_jabatan', 'rasio_pemenuhan_unor']
CRITERIA_N = len(AHP_CRITERIA_NAMES)

PAIRWISE_MATRIX = np.array([
    [1, 3, 5, 5, 2, 4, 3],
    [1/3, 1, 3, 3, 0.5, 3, 2],
    [0.2, 1/3, 1, 2, 1/3, 0.5, 1/3],
    [0.2, 1/3, 0.5, 1, 1/3, 1/3, 1/3],
    [1/2, 2, 3, 3, 1, 2, 2],
    [0.25, 1/3, 2, 3, 0.5, 1, 2],
    [1/3, 0.5, 3, 3, 0.5, 0.5, 1]
])

# Tipe Kriteria SAW
BENEFIT = ['kebutuhan_ideal', 'pensiun_5th', 'rasio_kebutuhan_unor', 'nilai_jabatan']
COST = ['eksisting_asn', 'eksisting_non_asn', 'rasio_pemenuhan_unor']

# --- FUNGSI BARU UNTUK SIMULASI ---

@st.cache_data
def load_data():
    """Hanya memuat dan membersihkan data, tanpa perhitungan SPK."""
    try:
        df = pd.read_csv("data_formasi_usulan.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def calculate_ahp(pairwise_matrix):
    """Menghitung Bobot AHP, CI, dan CR dari matriks input."""
    
    col_sum = pairwise_matrix.sum(axis=0)
    normalized_matrix = pairwise_matrix / col_sum
    weights = normalized_matrix.mean(axis=1)
    weights = weights / weights.sum()
    
    df_bobot = pd.DataFrame({'Kriteria': AHP_CRITERIA_NAMES, 'Bobot AHP': np.round(weights, 4)})

    # Uji Konsistensi
    lambda_max = np.mean((pairwise_matrix @ weights) / weights)
    CI = (lambda_max - CRITERIA_N) / (CRITERIA_N - 1)
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(CRITERIA_N, 1.49)
    CR = CI / RI
    
    return weights, df_bobot, CR, RI, CI

@st.cache_data
def run_spk_analysis(df_input: pd.DataFrame, ahp_weights: np.ndarray):
    """Menjalankan Perangkingan SAW menggunakan Bobot AHP yang diberikan."""
    
    df = df_input.copy()

    # --- SAW (Perangkingan) ---
    norm_df = df.copy()
    
    # Normalisasi SAW
    for c in AHP_CRITERIA_NAMES:
        if c in df.columns:
            if c in BENEFIT:
                max_val = df[c].max()
                norm_df[c] = df[c] / max_val if max_val != 0 else 0
            elif c in COST:
                min_val = df[c].min()
                norm_df[c] = df[c].apply(lambda x: min_val / x if x != 0 and min_val != 0 else 0)

    # Hitung Skor SAW (Vektor S)
    for i, c in enumerate(AHP_CRITERIA_NAMES):
        if c in norm_df.columns:
            # Menggunakan bobot yang telah dihitung (ahp_weights[i])
            norm_df[c] = norm_df[c] * ahp_weights[i]

    df['Skor_SAW'] = norm_df[AHP_CRITERIA_NAMES].sum(axis=1)

    # PERANGKINGAN AKHIR
    hasil = df.sort_values(
        by=['Skor_SAW', 'kebutuhan_ideal', 'pensiun_5th'], 
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    hasil.index.name = 'Ranking'
    hasil.index += 1
    hasil = hasil.reset_index()

    return hasil

# -----------------------------------------------------
# 🌐 NAVIGASI DAN INISIALISASI UTAMA
# -----------------------------------------------------

# 1. Muat Data Mentah
df_raw = load_data()

# 2. Tangani kasus File Not Found
if df_raw.empty:
    st.error("File 'data_formasi_usulan.csv' tidak ditemukan. Harap pastikan file data berada di direktori yang sama.")
    # Inisialisasi default agar tidak error di session state
    if 'initial_result' not in st.session_state:
        st.session_state['initial_result'] = {
            'bobot': pd.DataFrame({'Kriteria': AHP_CRITERIA_NAMES, 'Bobot AHP': 0.0}),
            'CR': 0.0, 'RI': 0.0, 'CI': 0.0, 'hasil_saw': pd.DataFrame()
        }
else:
    # 3. Hitung Hasil Awal (Initial) jika belum dihitung
    if 'initial_result' not in st.session_state:
        initial_weights, initial_bobot_df, initial_CR, initial_RI, initial_CI = calculate_ahp(PAIRWISE_MATRIX)
        df_hasil_initial = run_spk_analysis(df_raw, initial_weights)
        st.session_state['initial_result'] = {
            'bobot': initial_bobot_df,
            'CR': initial_CR,
            'RI': initial_RI,
            'CI': initial_CI,
            'hasil_saw': df_hasil_initial
        }
    
# Tentukan data hasil yang akan digunakan (jika data ada, gunakan yang awal)
df_hasil = st.session_state['initial_result']['hasil_saw']


menu = st.sidebar.radio(
    "📍 Pilih Halaman Analisis:",
    [
        "🏠 Overview Dashboard",
        "🏆 Hasil Perangkingan Formasi",
        "⚙️ Analisis Bobot Kriteria (AHP)",
        "🧪 Uji Sensitivitas (Spearman)"  # <--- Menu Baru
    ]
)

# -----------------------------------------------------
# 1. 🏠 OVERVIEW DASHBOARD 
# -----------------------------------------------------
if menu == "🏠 Overview Dashboard":
    st.markdown("<p class='title-main'>🏛️ IKHTISAR ANALISIS PRIORITAS FORMASI CASN</p>", unsafe_allow_html=True)
    
    if df_hasil.empty:
        st.warning("Data tidak tersedia untuk ditampilkan. Mohon periksa file `data_formasi_usulan.csv`.")
    else:
        st.markdown("Dashboard ini menyajikan hasil akhir pengusulan formasi berdasarkan prioritas tertinggi yang dihitung menggunakan metode gabungan **AHP-SAW**.")

        # --- PERHITUNGAN METRIK JFT-SW ---
        
        # Filter semua baris yang nama jabatannya mengandung awalan 'JFT-SW'
        df_jft_sw = df_hasil[df_hasil['kode_jabatan'].str.startswith('JFT-SW', na=False)].copy()
        
        total_usulan_jabatan = len(df_hasil)
        total_kekurangan_ideal = df_hasil['kebutuhan_ideal'].sum() if 'kebutuhan_ideal' in df_hasil.columns else 0
        
        # Metrik Khusus JFT-SW
        jft_sw_count = len(df_jft_sw)
        jft_sw_kekurangan = df_jft_sw['kebutuhan_ideal'].sum() if not df_jft_sw.empty else 0
        jft_sw_avg_score = df_jft_sw['Skor_SAW'].mean() if not df_jft_sw.empty else 0.0

        st.subheader("💡 Metrik Kunci & Fokus Strategis")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Total Jabatan Dianalisis", total_usulan_jabatan)
        c2.metric("🔥 Total Kekurangan Ideal", f"{int(total_kekurangan_ideal):,}")
        c3.metric("🏅 Jabatan Prioritas Tertinggi", df_hasil.iloc[0]['nama_jabatan'] if not df_hasil.empty else "N/A")
        c4.metric("🏢 OPD Prioritas Rata-rata Tertinggi", df_hasil.groupby('kode_unor')['Skor_SAW'].mean().nlargest(1).index[0] if 'kode_unor' in df_hasil.columns else "N/A")
        
        st.markdown("---")
        
        # TAMPILAN METRIK KHUSUS JABATAN STRATEGIS WILAYAH (JFT-SW)
        st.subheader("🚀 Fokus Jabatan Fungsional Teknis Strategis Wilayah (JFT-SW)")
        
        col_jft_metrik, col_jft_tabel = st.columns([1, 2])
        
        with col_jft_metrik:
            st.metric("Total Jenis JFT-SW (Kode Jabatan)", jft_sw_count)
            st.metric("Total Kekurangan JFT-SW", f"{int(jft_sw_kekurangan):,}")
        #    st.metric("Rata-rata Skor SAW JFT-SW", f"{jft_sw_avg_score:.4f}")

        with col_jft_tabel:
            st.markdown("##### Jabatan JFT-SW : ")
            if not df_jft_sw.empty:
                # Ambil 5 teratas berdasarkan Ranking/Skor SAW
                top_5_jft = df_jft_sw[['Ranking', 'nama_jabatan','Skor_SAW', 'kode_unor','unor']].head(5)
                
                st.dataframe(
                    top_5_jft.style.format({'Skor_SAW': "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Tidak ada data Jabatan JFT-SW yang ditemukan.")


        st.markdown("---")
        
        # --- GRAFIK ANALISIS ---
        st.subheader("📈 Visualisasi Hasil Kunci")

        col_grafik_1, col_grafik_2 = st.columns(2)

        # Grafik 1: 10 OPD dengan Rata-rata Skor SAW Tertinggi
        opd_prioritas = df_hasil.groupby('kode_unor')['Skor_SAW'].mean().nlargest(10).reset_index()
        with col_grafik_1:
            st.markdown("##### 10 Unit Organisasi (OPD) dengan Rata-rata Prioritas Tertinggi (Skor SAW)")
            if not opd_prioritas.empty:
                fig1 = px.bar(opd_prioritas, 
                            x='Skor_SAW', 
                            y='kode_unor', 
                            orientation='h', 
                            color='Skor_SAW',
                            color_continuous_scale=px.colors.sequential.Agsunset,
                            labels={'Skor_SAW': 'Rata-rata Skor SAW', 'kode_unor': 'Unit Organisasi'},
                            height=400)
                fig1.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig1, use_container_width=True)

        # Tabel Penjelas di Bawah Grafik untuk 10 OPD tersebut
                st.markdown("###### 📋 Keterangan Unit Organisasi (Top 10 OPD):")
                if 'unor' in df_hasil.columns:
                    # Ambil mapping nama unor khusus untuk 10 OPD teratas yang tampil di grafik
                    mapping_unor = df_hasil[['kode_unor', 'unor']].drop_duplicates()
                    df_tabel_penjelas = pd.merge(opd_prioritas[['kode_unor']], mapping_unor, on='kode_unor', how='left')
                    df_tabel_penjelas.columns = ['Kode Unor', 'Nama Unit Organisasi (OPD)']
                    
                    st.dataframe(
                        df_tabel_penjelas, 
                        use_container_width=True, 
                        hide_index=True
                    )
                else:
                    st.info("Kolom nama unit organisasi (misal: 'unor') tidak ditemukan di dalam data.")


        # Grafik 2: Distribusi Skor SAW berdasarkan Jenis Jabatan
        jenis_jab_dist = df_hasil.groupby('jenis_jabatan')['Skor_SAW'].mean().reset_index()
        with col_grafik_2:
            st.markdown("##### Prioritas Rata-rata Berdasarkan Jenis Jabatan")
            if not jenis_jab_dist.empty:
                fig2 = px.pie(jenis_jab_dist, 
                            values='Skor_SAW', 
                            names='jenis_jabatan', 
                            title='Proporsi Rata-rata Skor SAW',
                            color_discrete_sequence=px.colors.sequential.Blues_r)
                fig2.update_traces(textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)



# -----------------------------------------------------
# 2. 🏆 HASIL PERANGKINGAN FORMASI
# -----------------------------------------------------
elif menu == "🏆 Hasil Perangkingan Formasi":
    st.markdown("<p class='title-main'>🏆 TABEL HASIL PERANGKINGAN FORMASI CASN (AHP-SAW)</p>", unsafe_allow_html=True)
    
    if df_hasil.empty:
        st.warning("Data tidak tersedia untuk ditampilkan.")
    else:
        st.markdown("Tabel ini menyajikan daftar jabatan yang diusulkan, diurutkan berdasarkan **Skor SAW** tertinggi sebagai indikator prioritas.")
        
        # ... (Kode Filter dan Tampilan Tabel) ... (dihilangkan untuk fokus pada perbaikan)
        
        # --- FILTER DATA ---
        st.subheader("🗂️ Opsi Filter")
        
        col_filter_1, col_filter_2, col_filter_3 = st.columns(3)
        
        with col_filter_1:
            unor_options = sorted(df_hasil['unor'].unique().tolist()) if 'unor' in df_hasil.columns else []
            unor_select = st.multiselect("🏢 Filter Unit Organisasi (OPD)", options=unor_options, default=[])
        with col_filter_2:
            jenis_options = sorted(df_hasil['jenis_jabatan'].unique().tolist()) if 'jenis_jabatan' in df_hasil.columns else []
            jenis_select = st.multiselect("🧑‍💼 Filter Jenis Jabatan", options=jenis_options, default=[])
        with col_filter_3:
            top_n = st.slider("Lihat Top N Peringkat", 10, len(df_hasil), 50, step=10)

        filtered_df = df_hasil.copy()
        
        if unor_select:
            filtered_df = filtered_df[filtered_df['unor'].isin(unor_select)]
        if jenis_select:
            filtered_df = filtered_df[filtered_df['jenis_jabatan'].isin(jenis_select)]
        
        filtered_df = filtered_df.head(top_n)

        st.markdown("---")
        st.markdown(f"**Menampilkan {len(filtered_df)} dari {len(df_hasil)} Hasil Perangkingan**")

        # Kolom yang akan ditampilkan
        display_cols = ['Ranking', 'kode_unor', 'unor' ,'nama_jabatan', 'jenis_jabatan','abk' ,'kebutuhan_ideal',
                        'eksisting_non_asn','eksisting_asn', 'pensiun_5th', 'nilai_jabatan','rasio_pemenuhan_unor' ,'Skor_SAW']
        
        final_display_cols = [col for col in display_cols if col in filtered_df.columns]

        st.dataframe(
            filtered_df[final_display_cols].style.format({'Skor_SAW': "{:.4f}", 'kebutuhan_ideal': "{:,.0f}"}),
            use_container_width=True,
            hide_index=True
        )
        
        # Tombol Download
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(filtered_df)
        st.download_button(
            label="📥 Unduh Hasil Filter ke CSV",
            data=csv,
            file_name='hasil_perangkingan_formasi_CASN_filtered.csv',
            mime='text/csv',
        )


elif menu == "⚙️ Analisis Bobot Kriteria (AHP)":
    st.markdown("<p class='title-main'>⚙️ SIMULASI & ANALISIS BOBOT KRITERIA (METODE AHP)</p>", unsafe_allow_html=True)
    
    if df_raw.empty:
        st.warning("Data tidak tersedia untuk melakukan simulasi. Mohon periksa file `data_formasi_usulan.csv`.")
        st.stop()

    st.markdown("Gunakan tabel di bawah ini untuk memasukkan nilai **perbandingan berpasangan** baru dan mensimulasikan pengaruh bobot kriteria terhadap hasil akhir perangkingan.")

    # 1. INISIALISASI MATRIKS KE SESSION STATE (Agar input tersimpan permanen)
    if 'editable_ahp_matrix' not in st.session_state:
        st.session_state['editable_ahp_matrix'] = pd.DataFrame(
            PAIRWISE_MATRIX,
            index=AHP_CRITERIA_NAMES,
            columns=AHP_CRITERIA_NAMES
        )

    st.subheader("📝 Input Matriks Perbandingan Berpasangan")
    st.info("⚠️ **Panduan:** Hanya ubah nilai di **ATAS diagonal utama**. Nilai di bawah diagonal akan dihitung otomatis sebagai inversnya ($1/a_{ij}$).")

    # Display data editor terhubung langsung dengan session state
    edited_df = st.data_editor(
        st.session_state['editable_ahp_matrix'],
        use_container_width=True,
        key="ahp_editor_widget",
        column_config={
            col: st.column_config.NumberColumn(format="%.4f") for col in AHP_CRITERIA_NAMES
        }
    )

    col_btn1, col_btn2 = st.columns([2, 1])
    
    with col_btn1:
        btn_calculate = st.button("🔄 Hitung Ulang Bobot & Perangkingan", type="primary")
    with col_btn2:
        btn_reset = st.button("↩️ Reset ke Default")

    # Fitur Reset ke Nilai Awal
    if btn_reset:
        st.session_state['editable_ahp_matrix'] = pd.DataFrame(
            PAIRWISE_MATRIX,
            index=AHP_CRITERIA_NAMES,
            columns=AHP_CRITERIA_NAMES
        )
        st.session_state.pop('sim_result', None)
        st.rerun()

    # Fitur Hitung Ulang
    if btn_calculate:
        try:
            # Ambil data dari editor
            new_matrix = edited_df.values.copy().astype(float)
            
            # 2. PROSES ATURAN AHP (Diagonal = 1, Bawah = 1/Atas)
            np.fill_diagonal(new_matrix, 1.0)
            
            for i in range(CRITERIA_N):
                for j in range(i + 1, CRITERIA_N):
                    val_ij = new_matrix[i, j]
                    # Pastikan tidak terjadi pembagian dengan nol
                    new_matrix[j, i] = 1.0 / val_ij if val_ij > 0 else 0.0

            # Simpan matriks simetris yang baru kembali ke session state
            updated_df = pd.DataFrame(
                new_matrix,
                index=AHP_CRITERIA_NAMES,
                columns=AHP_CRITERIA_NAMES
            )
            st.session_state['editable_ahp_matrix'] = updated_df
            
            # 3. LAKUKAN PERHITUNGAN AHP & SAW BARU
            weights_sim, bobot_df_sim, CR_sim, RI_sim, CI_sim = calculate_ahp(new_matrix)
            df_hasil_sim = run_spk_analysis(df_raw, weights_sim)
            
            # Simpan hasil kalkulasi
            st.session_state['sim_result'] = {
                'bobot': bobot_df_sim,
                'CR': CR_sim,
                'RI': RI_sim,
                'CI': CI_sim,
                'hasil_saw': df_hasil_sim
            }
            
            st.success("✅ Perhitungan berhasil diperbarui!")
            st.rerun()

        except Exception as e:
            st.error(f"Gagal melakukan simulasi. Pastikan semua input numerik bernilai lebih besar dari 0. Error: {e}")

    # --- TAMPILKAN HASIL ---
    st.markdown("---")
    if 'sim_result' in st.session_state:
        results = st.session_state['sim_result']
        st.subheader("✨ Hasil Simulasi Bobot Baru")
    else:
        results = st.session_state['initial_result']
        st.subheader("📊 Hasil Bobot Awal (Default)")

    current_bobot = results['bobot']
    current_CR = results['CR']
    current_RI = results['RI']
    current_CI = results['CI']
    current_hasil_saw = results['hasil_saw']

    # Tampilkan Uji Konsistensi
    col_cr1, col_cr2, col_cr3 = st.columns(3)
    col_cr1.metric("Rasio Konsistensi (CR)", f"{current_CR:.4f}")
    col_cr2.metric("Indeks Konsistensi (CI)", f"{current_CI:.4f}") 
    col_cr3.metric("Random Index (RI)", f"{current_RI}")
    
    if current_CR < 0.1:
        st.success("✅ Matriks Konsisten (CR < 0.1). Bobot AHP Valid.")
    else:
        st.error("⚠️ Matriks TIDAK KONSISTEN (CR ≥ 0.1). Bobot AHP Perlu Direvisi!")

    # --- LOGIKA ANALISIS SENSITIVITAS (SPEARMAN) ---
    if 'initial_result' in st.session_state and 'sim_result' in st.session_state:
        df_awal = st.session_state['initial_result']['hasil_saw']
        df_baru = st.session_state['sim_result']['hasil_saw']

        # Pastikan urutan data sama berdasarkan kunci unik (misal: gabungan unor + jabatan)
        # Kita gunakan 'nama_jabatan' dan 'kode_unor' sebagai key perbandingan
        df_awal = df_awal.sort_values(['kode_unor', 'nama_jabatan'])
        df_baru = df_baru.sort_values(['kode_unor', 'nama_jabatan'])

        # Hitung Korelasi Spearman pada kolom Ranking
        coef, p_value = spearmanr(df_awal['Ranking'], df_baru['Ranking'])
        
        st.session_state['spearman_coef'] = coef
    
    # 6. Tampilkan Bobot dan Perangkingan
    col_bobot_1, col_bobot_2 = st.columns([1, 2])
    
    with col_bobot_1:
        st.subheader("Bobot Prioritas AHP")
        df_bobot_display = current_bobot.copy()
        df_bobot_display['Tipe'] = df_bobot_display['Kriteria'].apply(
            lambda x: 'Benefit' if x in BENEFIT else 'Cost'
        )
        st.dataframe(df_bobot_display.style.format({'Bobot AHP': "{:.4f}"}), hide_index=True, use_container_width=True)

    with col_bobot_2:
        st.subheader("Distribusi Bobot Kriteria")
        if not current_bobot.empty:
            fig_bobot = px.pie(
                current_bobot, 
                values='Bobot AHP', 
                names='Kriteria', 
                title='Kontribusi Bobot Kriteria Terhadap Skor SAW',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_bobot.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_bobot, use_container_width=True)

            
    # Tampilkan 5 Hasil Perangkingan Teratas dari Simulasi
    st.subheader("🏆 TOP 5 Hasil Perangkingan (Simulasi)")
    if not current_hasil_saw.empty:
        display_cols = ['Ranking', 'kode_unor','unor', 'nama_jabatan', 'kebutuhan_ideal', 'Skor_SAW']
        st.dataframe(
            current_hasil_saw[display_cols].head(5).style.format({'Skor_SAW': "{:.4f}", 'kebutuhan_ideal': "{:,.0f}"}),
            hide_index=True,
            use_container_width=True
        )

# -----------------------------------------------------
# 4. 🧪 UJI SENSITIVITAS (SPEARMAN) - INPUT BOBOT LANGSUNG
# -----------------------------------------------------
elif menu == "🧪 Uji Sensitivitas (Spearman)":
    st.markdown("<p class='title-main'>🧪 UJI SENSITIVITAS & ROBUSTITAS PERINGKAT</p>", unsafe_allow_html=True)
    
    if df_raw.empty:
        st.warning("Data tidak tersedia. Mohon periksa file `data_formasi_usulan.csv`.")
        st.stop()

    st.info("""
    **Metode:** Halaman ini digunakan untuk menguji seberapa stabil peringkat jabatan jika nilai bobot kriteria diubah secara manual. 
    Perubahan diukur menggunakan **Koefisien Korelasi Rank Spearman (ρ)**.
    """)

    # Ambil bobot default (dari perhitungan AHP awal)
    bobot_awal = st.session_state['initial_result']['bobot']['Bobot AHP'].values
    
    # --- INPUT AREA ---
    st.subheader("✍️ Masukkan Simulasi Bobot Baru")
    st.markdown("Sesuaikan nilai bobot di kolom **'Bobot Simulasi'**")
    #st.markdown("Sesuaikan nilai bobot di kolom **'Bobot Simulasi'**. Total bobot tidak harus 1 (sistem akan menormalisasi otomatis).")
    df_input_sens = pd.DataFrame({
        'Kriteria': AHP_CRITERIA_NAMES,
        'Bobot Awal (AHP)': bobot_awal,
        'Bobot Simulasi': bobot_awal  # Defaultnya sama dengan awal
    })

    edited_sens_df = st.data_editor(
        df_input_sens,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Kriteria": st.column_config.TextColumn(disabled=True),
            "Bobot Awal (AHP)": st.column_config.NumberColumn(format="%.4f", disabled=True),
            "Bobot Simulasi": st.column_config.NumberColumn(format="%.4f", min_value=0.000, step=0.0001)
        }
    )

    if st.button("📊 Hitung Korelasi Spearman", type="primary"):
        # Normalisasi
        raw_vals = edited_sens_df['Bobot Simulasi'].values
        sum_vals = np.sum(raw_vals)
        
        if sum_vals == 0:
            st.error("Total bobot tidak boleh nol!")
        else:
            norm_vals = raw_vals / sum_vals
            
            # Hitung SAW dengan bobot simulasi
            df_hasil_sim = run_spk_analysis(df_raw, norm_vals)
            df_awal = st.session_state['initial_result']['hasil_saw']

            # Gabungkan untuk hitung Spearman
            df_spearman = pd.merge(
                df_awal[['nama_jabatan', 'unor','kode_unor', 'Ranking']].rename(columns={'Ranking': 'Rank_Awal'}),
                df_hasil_sim[['nama_jabatan','unor', 'kode_unor', 'Ranking']].rename(columns={'Ranking': 'Rank_Simulasi'}),
                on=['nama_jabatan','unor', 'kode_unor']
            )

            # Hitung d_i dan d_i^2 (Logika seperti di Colab/Jurnal)
            df_spearman['d_i'] = df_spearman['Rank_Awal'] - df_spearman['Rank_Simulasi']
            df_spearman['d_i^2'] = df_spearman['d_i'] ** 2
            
            total_d2 = df_spearman['d_i^2'].sum()
            n = len(df_spearman)
            rho_s = 1 - ((6 * total_d2) / (n * (n**2 - 1)))

            # --- DISPLAY HASIL ---
            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric("Koefisien Spearman (ρ)", f"{rho_s:.4f}")
                if rho_s > 0.9:
                    st.success("✅ **Sangat Robust**")
                elif rho_s > 0.7:
                    st.info("⚠️ **Cukup Stabil**")
                else:
                    st.warning("🚨 **Sangat Sensitif**")

            with c2:
                # Grafik Perbandingan Bobot
                fig_bar = px.bar(
                    pd.DataFrame({
                        'Kriteria': AHP_CRITERIA_NAMES,
                        'Awal': bobot_awal,
                        'Simulasi': norm_vals
                    }).melt(id_vars='Kriteria'),
                    x='Kriteria', y='value', color='variable', barmode='group',
                    labels={'value': 'Nilai Bobot', 'variable': 'Skenario'},
                    height=300
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Tabel Detail (Top 10)
            st.subheader("📋 Tabel Perbandingan Peringkat (Top 10)")
            st.dataframe(
                df_spearman.sort_values('Rank_Awal').head(10).style.format({
                    'Rank_Awal': '{:,.0f}',
                    'Rank_Simulasi': '{:,.0f}',
                    'd_i': '{:,.0f}',
                    'd_i^2': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Keterangan Tambahan
            st.caption(f"Analisis dilakukan terhadap {n} baris data jabatan.")