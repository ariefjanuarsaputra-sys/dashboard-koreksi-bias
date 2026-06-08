import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================================================================
# KONFIGURASI HALAMAN
# =========================================================================
st.set_page_config(
    page_title="Dashboard Koreksi Bias Curah Hujan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# CSS PREMIUM - DESAIN PROFESIONAL & RESPONSIF MOBILE
# =========================================================================
st.markdown("""
<style>
    /* --- GLOBAL --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #f4f6f9;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8eaed;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* --- HEADER BANNER --- */
    .header-banner {
        background: linear-gradient(120deg, #0f2b4a 0%, #1a5298 55%, #2577cc 100%);
        border-radius: 14px;
        padding: 20px 26px;
        margin-bottom: 22px;
        color: white;
        box-shadow: 0 6px 24px rgba(21, 82, 152, 0.28);
    }
    .header-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
    }
    .header-title {
        font-size: 18px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.30);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .header-bottom-row {
        display: flex;
        align-items: center;
        gap: 6px;
        padding-top: 12px;
        border-top: 1px solid rgba(255,255,255,0.15);
        flex-wrap: wrap;
    }
    .header-chip {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        opacity: 0.92;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .header-chip-sep {
        opacity: 0.3;
        font-size: 14px;
        margin: 0 2px;
    }

    /* --- METRIC CARDS --- */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 18px;
        border-top: 4px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        position: relative;
    }
    .metric-card.blue   { border-top-color: #2980d4; }
    .metric-card.teal   { border-top-color: #1D9E75; }
    .metric-card.amber  { border-top-color: #EF9F27; }
    .metric-card.coral  { border-top-color: #D85A30; }
    .metric-card.purple { border-top-color: #7F77DD; }
    
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #7a8499;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-value.blue   { color: #1a6bbf; }
    .metric-value.teal   { color: #0e7d5c; }
    .metric-value.amber  { color: #b87a10; }
    .metric-value.coral  { color: #b84820; }
    .metric-value.purple { color: #5c52b8; }
    
    .metric-desc {
        font-size: 11px;
        color: #9ba3b4;
    }
    .metric-badge {
        position: absolute;
        top: 14px;
        right: 14px;
        font-size: 10px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 20px;
    }
    .badge-good   { background: #e6f7f1; color: #0e7d5c; }
    .badge-warn   { background: #fff8e6; color: #b87a10; }
    .badge-info   { background: #e8f2fc; color: #1a6bbf; }
    .badge-danger { background: #fdecea; color: #b84820; }

    /* --- SECTION CARD --- */
    .section-card {
        background: #ffffff;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 18px;
        overflow: hidden;
    }
    .section-header {
        padding: 14px 20px;
        border-bottom: 1px solid #f0f2f5;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .section-title {
        font-size: 14px;
        font-weight: 600;
        color: #1a2540;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .method-pill {
        font-size: 12px;
        background: #e6f7f1;
        color: #0e7d5c;
        border-radius: 20px;
        padding: 4px 14px;
        font-weight: 600;
        border: 1px solid #c2ebd9;
    }

    /* --- REGRESSION BANNER --- */
    .reg-banner {
        background: #e8f2fc;
        border: 1px solid #b8d8f5;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #1a4f8c;
        margin-bottom: 14px;
    }

    /* --- PANDUAN NILAI --- */
    .guide-item {
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    .guide-good   { background: #e6f7f1; color: #0e7d5c; }
    .guide-warn   { background: #fff8e6; color: #9a6b00; }
    .guide-danger { background: #fdecea; color: #a03020; }

    /* --- INFO TABLE (SIDEBAR) --- */
    .info-table {
        width: 100%;
        font-size: 12px;
        border-collapse: collapse;
    }
    .info-table tr {
        border-bottom: 1px solid #f0f2f5;
    }
    .info-table tr:last-child { border-bottom: none; }
    .info-table td { padding: 6px 2px; }
    .info-table td:first-child { color: #7a8499; }
    .info-table td:last-child  { font-weight: 600; text-align: right; }

    /* --- SIDEBAR SECTION HEADER --- */
    .sidebar-section {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: #9ba3b4;
        margin: 16px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #f0f2f5;
    }

    /* --- DOWNLOAD ROW --- */
    .dl-row {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .dl-info-title { font-size: 14px; font-weight: 600; color: #1a2540; }
    .dl-info-sub   { font-size: 12px; color: #9ba3b4; margin-top: 2px; }

    div[data-testid="stDownloadButton"] button {
        background: #f4f6f9 !important;
        border: 1px solid #dde1ea !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #1a2540 !important;
        padding: 8px 16px !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: #e8eaed !important;
    }

    div[data-testid="stMetric"] {
        display: none !important;
    }

    @media (max-width: 768px) {
        .metrics-row {
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .header-top-row {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }
        .header-title {
            font-size: 16px;
        }
        .header-bottom-row {
            gap: 4px;
        }
        .header-chip {
            font-size: 11px;
            padding: 4px 10px;
        }
        .header-chip-sep {
            display: none;
        }
        .reg-banner {
            font-size: 11px;
            line-height: 1.4;
        }
        .metric-badge {
            display: none;
        }
        .metric-value {
            font-size: 22px;
        }
        .section-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 6px;
        }
    }

    @media (max-width: 480px) {
        .metrics-row {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# FUNGSI METRIK STATISTIK & INDEKS EKSTREM
# =========================================================================
def calculate_nse(obs, sim):
    pembagi = np.sum((obs - np.mean(obs)) ** 2)
    return np.nan if pembagi == 0 else 1 - (np.sum((obs - sim) ** 2) / pembagi)

def calculate_mbe(obs, sim):
    return np.mean(sim - obs)

def nse_badge(nse):
    if np.isnan(nse): return "badge-info", "N/A"
    if nse > 0.7:     return "badge-good", "Sangat Baik"
    if nse > 0.4:     return "badge-warn", "Cukup"
    return "badge-danger", "Perlu Perbaikan"

def mbe_badge(mbe):
    if abs(mbe) < 1: return "badge-good", "Minim Bias"
    if abs(mbe) < 3: return "badge-warn", f"{'Overestimate' if mbe > 0 else 'Underestimate'}"
    return "badge-danger", f"{'Overestimate' if mbe > 0 else 'Underestimate'}"

def calculate_etccdi_indices(series, dates):
    df_temp = pd.DataFrame({'rr': series, 'Tanggal': dates})
    df_temp['Tahun'] = df_temp['Tanggal'].dt.year
    
    annual_indices = []
    for yr, group in df_temp.groupby('Tahun'):
        rr_vals = group['rr'].to_numpy()
        rx1day = np.nanmax(rr_vals) if len(rr_vals) > 0 else np.nan
        
        cdd_max, cwd_max = 0, 0
        current_cdd, current_cwd = 0, 0
        for val in rr_vals:
            if np.isnan(val): continue
            if val < 1.0:
                current_cdd += 1
                cwd_max = max(cwd_max, current_cwd)
                current_cwd = 0
            else:
                current_cwd += 1
                cdd_max = max(cdd_max, current_cdd)
                current_cdd = 0
        cdd_max = max(cdd_max, current_cdd)
        cwd_max = max(cwd_max, current_cwd)
        annual_indices.append({'Tahun': yr, 'Rx1day': rx1day, 'CDD': cdd_max, 'CWD': cwd_max})
    return pd.DataFrame(annual_indices)

def compute_empirical_cdf(data):
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0: return np.array([]), np.array([])
    sorted_data = np.sort(clean_data)
    y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y_values


# =========================================================================
# OPERASI MATEMATIKA NATIVE NUMPY (KOREKSI BIAS)
# =========================================================================
def apply_linear_scaling(obs, model, months):
    corrected = model.copy()
    for m in range(1, 13):
        idx = (months == m)
        if np.sum(idx) > 0 and np.mean(model[idx]) > 0:
            corrected[idx] = model[idx] * (np.mean(obs[idx]) / np.mean(model[idx]))
    return np.clip(corrected, 0.0, None)

def apply_variance_scaling(obs, model, months):
    corrected = model.copy()
    for m in range(1, 13):
        idx = (months == m)
        if np.sum(idx) > 0 and np.std(model[idx]) > 0:
            corrected[idx] = np.mean(obs[idx]) + (model[idx] - np.mean(model[idx])) * (np.std(obs[idx]) / np.std(model[idx]))
    return np.clip(corrected, 0.0, None)

def apply_quantile_mapping(obs, model, months):
    corrected = model.copy()
    for m in range(1, 13):
        idx = (months == m)
        if np.sum(idx) > 0:
            percentiles = np.array([np.percentile(model[idx], p) for p in range(101)])
            obs_values = np.array([np.percentile(obs[idx], p) for p in range(101)])
            corrected[idx] = np.interp(model[idx], percentiles, obs_values)
    return np.clip(corrected, 0.0, None)

def apply_detrended_quantile_mapping(obs, model, months):
    corrected = model.copy()
    for m in range(1, 13):
        idx = (months == m)
        if np.sum(idx) > 0:
            m_mod = model[idx]
            x = np.arange(len(m_mod))
            p = np.polyfit(x, m_mod, 1)
            trend = np.polyval(p, x)
            m_mod_detrend = m_mod - trend + np.mean(m_mod)
            percentiles = np.array([np.percentile(m_mod_detrend, p) for p in range(101)])
            obs_values = np.array([np.percentile(obs[idx], p) for p in range(101)])
            m_mod_mapped = np.interp(m_mod_detrend, percentiles, obs_values)
            corrected[idx] = m_mod_mapped + trend - np.mean(m_mod)
    return np.clip(corrected, 0.0, None)

def apply_quantile_delta_mapping(obs, model, months):
    corrected = model.copy()
    for m in range(1, 13):
        idx = (months == m)
        if np.sum(idx) > 0:
            m_mod = model[idx]
            ranks = (np.argsort(np.argsort(m_mod)) / (len(m_mod) - 1)) * 100
            obs_quantiles = np.array([np.percentile(obs[idx], r) for r in ranks])
            mod_quantiles = np.array([np.percentile(m_mod, r) for r in ranks])
            delta = np.where(mod_quantiles > 0, m_mod / mod_quantiles, 1.0)
            corrected[idx] = obs_quantiles * delta
    return np.clip(corrected, 0.0, None)

def calculate_metrics(obs, simulated):
    mask = ~np.isnan(obs) & ~np.isnan(simulated)
    o, s = obs[mask], simulated[mask]
    if len(o) == 0: return 0, 0, 0, 0
    mae = np.mean(np.abs(s - o))
    rmse = np.sqrt(np.mean((s - o) ** 2))
    mbe = np.mean(s - o)
    den = np.sum((o - np.mean(o)) ** 2)
    nse = 1 - (np.sum((o - s) ** 2) / den) if den != 0 else 0
    return mae, rmse, mbe, nse

def get_performance_status_by_rmse(rmse_metode, rmse_raw):
    if rmse_metode > rmse_raw: return "Kurang Bagus (Error Meningkat)"
    pct = ((rmse_raw - rmse_metode) / rmse_raw) * 100
    if pct > 50: return f"Sangat Bagus (RMSE Turun {pct:.1f}%)"
    if 25 <= pct <= 50: return f"Bagus (RMSE Turun {pct:.1f}%)"
    if 0 < pct < 25: return f"Cukup Bagus (RMSE Turun {pct:.1f}%)"
    return "Tidak Ada Perubahan"


# =========================================================================
# ENGINE UTAMA PENYEDIA DATA (TETAP UTUH & KONSISTEN)
# =========================================================================
@st.cache_data
def load_data(faktor_delta):
    df_obs = pd.read_excel("Minang_obs 1991-2025.xlsx")
    df_model = pd.read_excel("Book1.xlsx")
    
    # Standardisasi penamaan kolom utama
    df_obs.rename(columns={'RR': 'Observasi'}, inplace=True)
    df_model.rename(columns={'RR': 'Model_Raw'}, inplace=True)
    
    df_obs['Observasi'] = pd.to_numeric(df_obs['Observasi'], errors='coerce')
    df_model['Model_Raw'] = pd.to_numeric(df_model['Model_Raw'], errors='coerce')
    
    df_obs['Tanggal'] = pd.to_datetime(df_obs['Tanggal'])
    df_model['Tanggal'] = pd.to_datetime(df_model['Tanggal'])
    
    # PERBAIKAN FATAL: Menggunakan left join agar data model (1991-2025) tidak terbuang!
    res_df = pd.merge(df_model, df_obs, on='Tanggal', how='left')
    
    # Mengisi kolom awal dengan data mentah agar tidak terjadi KeyError di bagian bawah skrip
    res_df['Delta_Method'] = res_df['Model_Raw'].copy()
    res_df['Linear_Scaling'] = res_df['Model_Raw'].copy()
    res_df['Variance_Scaling'] = res_df['Model_Raw'].copy()
    res_df['Quantile_Mapping'] = res_df['Model_Raw'].copy()
    res_df['Detrended_Quantile_Mapping'] = res_df['Model_Raw'].copy()
    res_df['Quantile_Delta_Mapping'] = res_df['Model_Raw'].copy()

    # Hitung matematika array hanya pada baris di mana data observasi dan model tersedia
    mask = res_df['Observasi'].notna() & res_df['Model_Raw'].notna()
    if np.sum(mask) > 0:
        o_c = res_df.loc[mask, 'Observasi'].to_numpy().astype(float)
        m_c = res_df.loc[mask, 'Model_Raw'].to_numpy().astype(float)
        mo_c = res_df.loc[mask, 'Tanggal'].dt.month.to_numpy()
        
        # Hitung Delta Method
        h_delta = m_c.copy()
        for b in range(1, 13):
            mb = mo_c == b
            if np.sum(mb) > 0:
                h_delta[mb] = m_c[mb] + ((np.mean(o_c[mb]) - np.mean(m_c[mb])) * faktor_delta)
                
        res_df.loc[mask, 'Delta_Method'] = np.clip(h_delta, 0.0, None)
        res_df.loc[mask, 'Linear_Scaling'] = apply_linear_scaling(o_c, m_c, mo_c)
        res_df.loc[mask, 'Variance_Scaling'] = apply_variance_scaling(o_c, m_c, mo_c)
        res_df.loc[mask, 'Quantile_Mapping'] = apply_quantile_mapping(o_c, m_c, mo_c)
        res_df.loc[mask, 'Detrended_Quantile_Mapping'] = apply_detrended_quantile_mapping(o_c, m_c, mo_c)
        res_df.loc[mask, 'Quantile_Delta_Mapping'] = apply_quantile_delta_mapping(o_c, m_c, mo_c)
        
    return res_df


# =========================================================================
# KONTROL & KUSTOMISASI SIDEBAR
# =========================================================================
if 'delta_slider' not in st.session_state:
    st.session_state['delta_slider'] = 0.6

with st.sidebar:
    st.markdown('<div class="sidebar-section">⚙️ Metode Koreksi</div>', unsafe_allow_html=True)
    metode = st.selectbox(
        "Pilih Metode:",
        ["Sebelum Koreksi (ERA5)", "Delta Method", "Linear Scaling", "Variance Scaling", "Quantile Mapping", "Detrended Quantile Mapping", "Quantile Delta Mapping", "Perbandingan Semua Metode"],
        label_visibility="collapsed"
    )
    if metode == "Delta Method":
        st.session_state['delta_slider'] = st.slider("Faktor Bobot Delta:", 0.0, 1.0, st.session_state['delta_slider'], 0.1)

# Ambil data utama dari cache memory engine
df = load_data(st.session_state['delta_slider'])

with st.sidebar:
    st.markdown('<div class="sidebar-section">📊 Skala Evaluasi Metrik</div>', unsafe_allow_html=True)
    skala_evaluasi = st.radio("Pilih Skala Waktu:", ["Harian murni", "Bulanan (Akumulasi)"])
    
    st.markdown('<div class="sidebar-section">📅 Rentang Waktu Grafik</div>', unsafe_allow_html=True)
    min_date, max_date = df['Tanggal'].min().date(), df['Tanggal'].max().date()
    col_s1, col_s2 = st.columns(2)
    with col_s1: start_date = st.date_input("Dari", value=min_date, min_value=min_date, max_value=max_date)
    with col_s2: end_date = st.date_input("Sampai", value=max_date, min_value=min_date, max_value=max_date)

    # KEMBALI DISEDIAKAN UTUH: Seluruh Metadata Info Geografis Stasiun Anda yang sempat hilang
    st.markdown('<div class="sidebar-section">📍 Info Stasiun</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table class="info-table">
        <tr><td>Nama Stasiun</td><td>Minangkabau</td></tr>
        <tr><td>Kabupaten</td><td>Padang Pariaman</td></tr>
        <tr><td>Elevasi</td><td>3 m dpl</td></tr>
        <tr><td>Lintang</td><td>-0.7866°</td></tr>
        <tr><td>Bujur</td><td>100.2813°</td></tr>
        <tr><td>Total Data</td><td>{len(df):,} hari</td></tr>
        <tr><td>Data Hilang</td><td style="color:#b84820">{df['Observasi'].isna().sum()} hari</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">💡 Panduan Evaluasi Statistik</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="guide-item guide-good">📉 jika RMSE &amp; MSE Turun &nbsp;→&nbsp; Akurasi meningkat</div>
    <div class="guide-item guide-good">📉 jika MAE Mengecil &nbsp;→&nbsp; Deviasi model makin dekat ke observasi</div>
    <div class="guide-item guide-warn">🎯 jika MBE Mendekati 0.000 &nbsp;→&nbsp; Kebiasan hilang</div>
    <div class="guide-item guide-purple" style="background:#f0eefc; color:#5c52b8; border-radius:8px; padding:8px 12px; font-size:12px; margin-bottom:6px; font-weight:500;">📈 jika NSE &gt; 0.70 &nbsp;→&nbsp; Performa Sangat Baik</div>
    """, unsafe_allow_html=True)


# =========================================================================
# RENDERING HALAMAN DASHBOARD UTAMA
# =========================================================================
if metode == "Perbandingan Semua Metode":
    # ---------------------------------------------------------------------
    # MODE MULTIMETODE (7 GARIS)
    # ---------------------------------------------------------------------
    st.markdown(f"""<div class="header-banner"><div class="header-top-row"><p class="header-title">🌦️ Dashboard Analisis Komparatif Semua Metode Koreksi Bias</p><span class="header-badge" style="background:#1D9E75;">✅ Mode Komparasi Aktif (7 Metode)</span></div><div class="header-bottom-row"><span class="header-chip">📍 Stasiun Meteorologi Minangkabau</span><span class="header-chip-sep">·</span><span class="header-chip">📅 1991 – 2025</span><span class="header-chip-sep">·</span><span class="header-chip">🗂️ {len(df):,} titik data</span></div></div>""", unsafe_allow_html=True)
    st.markdown("### 📊 Tabel Perbandingan Performa Statistik (7 Metode)")
    
    list_metode = {
        "Sebelum Koreksi (ERA5)": df['Model_Raw'], "Delta Method": df['Delta_Method'], "Linear Scaling": df['Linear_Scaling'],
        "Variance Scaling": df['Variance_Scaling'], "Quantile Mapping": df['Quantile_Mapping'],
        "Detrended Quantile Mapping": df['Detrended_Quantile_Mapping'], "Quantile Delta Mapping": df['Quantile_Delta_Mapping']
    }
    
    v_obs = df.dropna(subset=['Observasi'])['Observasi'].values
    v_raw = df.dropna(subset=['Observasi'])['Model_Raw'].values
    _, rmse_raw, _, _ = calculate_metrics(v_obs, v_raw)
    
    rows = []
    for nama, s_data in list_metode.items():
        v_sim = s_data.loc[df['Observasi'].notna()].values
        mae, rmse, mbe, nse = calculate_metrics(v_obs, v_sim)
        rows.append({"Metode Koreksi Bias": nama, "MAE": round(mae, 3), "MSE": round(rmse**2, 3), "RMSE": round(rmse, 3), "MBE (Bias)": round(mbe, 3), "NSE": round(nse, 3), "Keterangan Performa": get_performance_status_by_rmse(rmse, rmse_raw)})
        
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    tab_cdf, tab_ts = st.tabs(["Kurva Distribusi CDF (7 Garis)", "Grafik Deret Waktu / Time Series (7 Garis)"])
    with tab_cdf:
        fig_cdf, ax_cdf = plt.subplots(figsize=(11, 5.5))
        obs_sorted = np.sort(v_obs)
        ax_cdf.plot(obs_sorted, np.arange(1, len(obs_sorted) + 1) / len(obs_sorted), label="Observasi BMKG", color="black", linewidth=3.0)
        for (nama, s_data), color in zip(list_metode.items(), ['red', 'blue', 'orange', 'green', 'purple', 'brown', 'magenta']):
            v_s = s_data.loc[df['Observasi'].notna()].values
            sim_s = np.sort(v_s)
            ax_cdf.plot(sim_s, np.arange(1, len(sim_s) + 1) / len(sim_s), label=nama, color=color, linestyle="--" if nama == "Sebelum Koreksi (ERA5)" else "-", linewidth=1.6)
        ax_cdf.set_xlim(0, 120)
        ax_cdf.legend(loc="lower right", fontsize=9)
        ax_cdf.grid(True, alpha=0.3)
        st.pyplot(fig_cdf)
        
    with tab_ts:
        fig_ts, ax_ts = plt.subplots(figsize=(12, 5))
        sample_df = df.head(100)
        ax_ts.plot(sample_df['Tanggal'], sample_df['Observasi'], label="Observasi BMKG", color="black", linewidth=2.8)
        for (nama, s_data), color in zip(list_metode.items(), ['red', 'blue', 'orange', 'green', 'purple', 'brown', 'magenta']):
            ax_ts.plot(sample_df['Tanggal'], s_data.head(100), label=nama, color=color, linestyle="--" if nama == "Sebelum Koreksi (ERA5)" else "-", alpha=0.7)
        ax_ts.legend(loc="upper right", fontsize=8, ncol=2)
        ax_ts.grid(True, alpha=0.3)
        st.pyplot(fig_ts)

    st.markdown("---")
    df_download = df[['Tanggal', 'Observasi', 'Model_Raw', 'Delta_Method', 'Linear_Scaling', 'Variance_Scaling', 'Quantile_Mapping', 'Detrended_Quantile_Mapping', 'Quantile_Delta_Mapping']].copy()
    col_dl1, col_dl2 = st.columns([3, 1])
    with col_dl1: st.markdown('<div class="dl-info-title">💾 Ekspor Hasil Analisis Multimetode</div><div class="dl-info-sub">Ekspor gabungan seluruh data koreksi bias.</div>', unsafe_allow_html=True)
    with col_dl2: st.download_button(label="📥 Unduh CSV (Semua Metode)", data=df_download.to_csv(index=False).encode('utf-8'), file_name="Perbandingan_Semua_Metode_Koreksi_Bias.csv", mime="text/csv", use_container_width=True)
    with st.expander("👁️ Klik untuk melihat intipan tabel data gabungan"): st.dataframe(df_download.head(200), use_container_width=True, height=320)

else:
  # ---------------------------------------------------------------------
    # MODE SELEKSI TUNGGAL
    # ---------------------------------------------------------------------
    if metode == "Sebelum Koreksi (ERA5)":
        df['Terprediksi'] = df['Model_Raw'].copy()
        text_label_hujau = "Metode: Sebelum Koreksi (ERA5)"
    else:
        col_name = metode.replace(' ', '_')
        df['Terprediksi'] = df[col_name].copy() if col_name in df.columns else df['Model_Raw'].copy()
        text_label_hujau = f"Setelah Koreksi ({metode})"

    mask_m = df['Observasi'].notna() & df['Terprediksi'].notna()
    df_v = df[mask_m].copy()

    if skala_evaluasi == "Bulanan (Akumulasi)":
        df_v.set_index('Tanggal', inplace=True)
        m_calc = df_v[['Observasi', 'Terprediksi']].resample('ME').sum()
        o_m, s_m = m_calc['Observasi'], m_calc['Terprediksi']
        lbl_b, lbl_mse = "mm/bulan", "mm²"
    else:
        o_m, s_m = df_v['Observasi'], df_v['Terprediksi']
        lbl_b, lbl_mse = "mm/hari", "mm²"

    mae, rmse, mbe, nse = (mean_absolute_error(o_m, s_m), np.sqrt(mean_squared_error(o_m, s_m)), calculate_mbe(o_m, s_m), calculate_nse(o_m, s_m)) if len(o_m) > 0 else (0,0,0,0)
    n_c, n_l = nse_badge(nse)
    m_c, m_l = mbe_badge(mbe)

    st.markdown(f"""<div class="header-banner"><div class="header-top-row"><p class="header-title">🌦️ Dashboard Analisis &amp; Koreksi Bias Curah Hujan</p><span class="header-badge">✅ {metode} ({skala_evaluasi})</span></div><div class="header-bottom-row"><span class="header-chip">📍 Stasiun Meteorologi Minangkabau</span><span class="header-chip-sep">·</span><span class="header-chip">📅 1991 – 2025</span><span class="header-chip-sep">·</span><span class="header-chip">🗂️ {len(df):,} titik data</span></div></div>""", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card blue">
            <div class="metric-label">MAE</div>
            <div class="metric-value blue">{mae:.3f}</div>
            <div class="metric-desc">Mean Absolute Error</div>
            <span class="metric-badge badge-info">{lbl_b}</span>
        </div>
        <div class="metric-card teal">
            <div class="metric-label">MSE</div>
            <div class="metric-value teal">{rmse**2:.3f}</div>
            <div class="metric-desc">Mean Squared Error</div>
            <span class="metric-badge badge-info">{lbl_mse}</span>
        </div>
        <div class="metric-card amber">
            <div class="metric-label">RMSE</div>
            <div class="metric-value amber">{rmse:.3f}</div>
            <div class="metric-desc">Root Mean Squared Error</div>
            <span class="metric-badge badge-warn">{lbl_b}</span>
        </div>
        <div class="metric-card coral">
            <div class="metric-label">MBE</div>
            <div class="metric-value coral">{mbe:.3f}</div>
            <div class="metric-desc">Mean Bias Error</div>
            <span class="metric-badge {m_c}">{m_l}</span>
        </div>
        <div class="metric-card purple">
            <div class="metric-label">NSE</div>
            <div class="metric-value purple">{nse:.3f}</div>
            <div class="metric-desc">Nash-Sutcliffe Efficiency</div>
            <span class="metric-badge {n_c}">{n_l}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_filtered = df[(df['Tanggal'].dt.date >= start_date) & (df['Tanggal'].dt.date <= end_date)].copy()
    df_filtered['Bulan'] = df_filtered['Tanggal'].dt.month
    monthly_avg = df_filtered.groupby('Bulan')[['Observasi', 'Model_Raw', 'Terprediksi']].mean().reset_index()

    st.markdown(f"""<div class="section-header" style="background:#fff;border-radius:14px 14px 0 0;padding:14px 20px;border-bottom:1px solid #f0f2f5;display:flex;align-items:center;justify-content:space-between;box-shadow:0 2px 10px rgba(0,0,0,0.06);"><span class="section-title">📈 Visualisasi Perbandingan &amp; Validasi Ekstrem</span><span class="method-pill">{text_label_hujau}</span></div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📉 Grafik Deret Waktu", "📊 Klimatologi Bulanan", "🎯 Scatter Plot 1:1", "⚡ Indeks Ekstrem Klimatologi", "📈 Kurva Distribusi CDF"])
    
    # Menentukan teks label dinamis berdasarkan pilihan di sidebar
    label_sumbu_y = "Curah Hujan (mm/bulan)" if skala_evaluasi == "Bulanan (Akumulasi)" else "Curah Hujan (mm/hari)"
    label_sumbu_x_ts = "Periode Waktu (Tanggal)" if skala_evaluasi != "Bulanan (Akumulasi)" else "Periode Waktu (Bulan-Tahun)"

    # ---------------------------------------------------------------------
    # TAB 1: GRAFIK DERET WAKTU (TIME SERIES)
    # ---------------------------------------------------------------------
    with tab1:
        fig_ts = go.Figure()
        if skala_evaluasi == "Bulanan (Akumulasi)":
            df_ts_m = df_filtered.set_index('Tanggal')[['Observasi', 'Terprediksi']].resample('ME').sum().reset_index()
            fig_ts.add_trace(go.Scatter(x=df_ts_m['Tanggal'], y=df_ts_m['Observasi'], mode='lines+markers', name='Observasi BMKG', line=dict(color='#2980d4')))
            fig_ts.add_trace(go.Scatter(x=df_ts_m['Tanggal'], y=df_ts_m['Terprediksi'], mode='lines+markers', name='ERA5 Mentah' if metode == "Sebelum Koreksi (ERA5)" else f'{metode}', line=dict(color='#D62728', dash='dot')))
        else:
            fig_ts.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Observasi'], mode='lines', name='Observasi BMKG', line=dict(color='#2980d4')))
            fig_ts.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Terprediksi'], mode='lines', name='ERA5 Mentah' if metode == "Sebelum Koreksi (ERA5)" else f'{metode}', line=dict(color='#D62728', dash='dot')))
        
        fig_ts.update_layout(
            hovermode="x unified", 
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            xaxis_title=label_sumbu_x_ts,
            yaxis_title=label_sumbu_y,
            margin=dict(t=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 2: KLIMATOLOGI BULANAN
    # ---------------------------------------------------------------------
    with tab2:
        fig_clim = go.Figure()
        fig_clim.add_trace(go.Scatter(x=monthly_avg['Bulan'], y=monthly_avg['Observasi'], mode='lines+markers', name='Observasi BMKG', line=dict(color='#2980d4', width=3)))
        if metode == "Sebelum Koreksi (ERA5)":
            fig_clim.add_trace(go.Scatter(x=monthly_avg['Bulan'], y=monthly_avg['Model_Raw'], mode='lines+markers', name='ERA5 Sebelum Koreksi', line=dict(color='#D62728', width=2.5, dash='dot')))
        else:
            fig_clim.add_trace(go.Scatter(x=monthly_avg['Bulan'], y=monthly_avg['Model_Raw'], mode='lines+markers', name='ERA5 Sebelum Koreksi', line=dict(color='#D62728', width=2, dash='dot')))
            fig_clim.add_trace(go.Scatter(x=monthly_avg['Bulan'], y=monthly_avg['Terprediksi'], mode='lines+markers', name=f'Setelah Koreksi ({metode})', line=dict(color='#1D9E75', width=2.5, dash='dash')))
        
        fig_clim.update_layout(
            xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Ags','Sep','Okt','Nov','Des']), 
            xaxis_title="Bulan Kalender",
            yaxis_title="Rata-rata Curah Hujan (mm/hari)",
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            margin=dict(t=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_clim, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 3: SCATTER PLOT 1:1 & REGRESI
    # ---------------------------------------------------------------------
    with tab3:
        if len(o_m) > 1:
            x_v, y_v = o_m.to_numpy().astype(float), s_m.to_numpy().astype(float)
            m, c = np.polyfit(x_v, y_v, 1)
            st.markdown(f"""<div class="reg-banner">📈 <strong>Analisis Regresi ({skala_evaluasi}):</strong> y = {m:.3f}x {"+" if c>=0 else "-"} {abs(c):.3f} &nbsp;|&nbsp; R² = {np.corrcoef(x_v, y_v)[0,1]**2:.3f}</div>""", unsafe_allow_html=True)
            max_v = max(x_v.max(), y_v.max()) * 1.05
            
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=x_v, y=y_v, mode='markers', name='Titik Data', marker=dict(color='rgba(41, 128, 212, 0.4)')))
            fig_sc.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines', name='Ideal 1:1', line=dict(color='#D85A30', dash='dash')))
            fig_sc.add_trace(go.Scatter(x=np.linspace(0, max_v, 100), y=m*np.linspace(0, max_v, 100), mode='lines', name='Garis Regresi', line=dict(color='#1D9E75')))
            
            fig_sc.update_layout(
                xaxis_title=f"Observasi BMKG ({lbl_b.split('/')[1]})",
                yaxis_title=f"Estimasi ERA5 ({lbl_b.split('/')[1]})",
                height=380, 
                plot_bgcolor='white', 
                paper_bgcolor='white',
                margin=dict(t=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 4: INDEKS EKSTREM KLIMATOLOGI (ETCCDI)
    # ---------------------------------------------------------------------
    with tab4:
        if len(df_filtered) > 365:
            ind_obs = calculate_etccdi_indices(df_filtered['Observasi'], df_filtered['Tanggal'])
            ind_raw = calculate_etccdi_indices(df_filtered['Model_Raw'], df_filtered['Tanggal'])
            c_idx = st.selectbox("Pilih Indeks Ekstrem:", ["Rx1day (Hujan Ekstrem Harian)", "CDD (Consecutive Dry Days)", "CWD (Consecutive Wet Days)"])
            c_key = 'Rx1day' if "Rx1day" in c_idx else ('CDD' if "CDD" in c_idx else 'CWD')
            
            # Unit dinamis sumbu Y untuk parameter ekstrem ekstrem
            y_ext_title = "Curah Hujan Maksimum (mm/hari)" if c_key == "Rx1day" else "Jumlah Hari Berurutan (Hari)"

            fig_ext = go.Figure()
            fig_ext.add_trace(go.Bar(x=ind_obs['Tahun'], y=ind_obs[c_key], name="Observasi BMKG", marker_color='#2980d4'))
            if metode == "Sebelum Koreksi (ERA5)":
                fig_ext.add_trace(go.Bar(x=ind_raw['Tahun'], y=ind_raw[c_key], name="ERA5 Sebelum Koreksi", marker_color='#D62728'))
            else:
                ind_cor = calculate_etccdi_indices(df_filtered['Terprediksi'], df_filtered['Tanggal'])
                fig_ext.add_trace(go.Bar(x=ind_raw['Tahun'], y=ind_raw[c_key], name="ERA5 Sebelum Koreksi", marker_color='#D62728'))
                fig_ext.add_trace(go.Bar(x=ind_cor['Tahun'], y=ind_cor[c_key], name=f"Setelah Koreksi ({metode})", marker_color='#1D9E75'))
            
            fig_ext.update_layout(
                barmode='group', 
                xaxis_title="Tahun Analisis",
                yaxis_title=y_ext_title,
                height=380, 
                plot_bgcolor='white', 
                paper_bgcolor='white',
                margin=dict(t=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_ext, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 5: KURVA DISTRIBUSI CDF
    # ---------------------------------------------------------------------
    with tab5:
        x_obs, y_obs = compute_empirical_cdf(df_filtered['Observasi'].to_numpy())
        x_raw, y_raw = compute_empirical_cdf(df_filtered['Model_Raw'].to_numpy())
        if len(x_obs) > 0 and len(x_raw) > 0:
            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(x=x_obs, y=y_obs, mode='lines', name='Observasi BMKG', line=dict(color='#2980d4', width=3)))
            if metode == "Sebelum Koreksi (ERA5)":
                fig_cdf.add_trace(go.Scatter(x=x_raw, y=y_raw, mode='lines', name='ERA5 Sebelum Koreksi', line=dict(color='#D62728', width=2.5, dash='dot')))
            else:
                x_cor, y_cor = compute_empirical_cdf(df_filtered['Terprediksi'].to_numpy())
                fig_cdf.add_trace(go.Scatter(x=x_raw, y=y_raw, mode='lines', name='ERA5 Sebelum Koreksi', line=dict(color='#D62728', width=2, dash='dot')))
                fig_cdf.add_trace(go.Scatter(x=x_cor, y=y_cor, mode='lines', name=f'Setelah Koreksi ({metode})', line=dict(color='#1D9E75', width=2.5, dash='dash')))
            
            fig_cdf.update_layout(
                height=400, 
                plot_bgcolor='white', 
                paper_bgcolor='white', 
                xaxis=dict(range=[0, 120]),
                xaxis_title="Nilai Curah Hujan (mm)",
                yaxis_title="Probabilitas Kumulatif - P(X ≤ x)",
                margin=dict(t=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_cdf, use_container_width=True)

    # ---------------------------------------------------------------------
    # EXPORT DATA DOWNLOAD
    # ---------------------------------------------------------------------
    st.markdown("---")
    if metode == "Sebelum Koreksi (ERA5)":
        df_final_export = df_filtered[['Tanggal', 'Observasi', 'Model_Raw']].copy()
        df_final_export.rename(columns={'Model_Raw': 'ERA5 Sebelum Koreksi'}, inplace=True)
        sub_text, f_name = "Kolom: Tanggal · Observasi · ERA5 Sebelum Koreksi", "ERA5_Sebelum_Koreksi_Murni.csv"
    else:
        df_final_export = df_filtered[['Tanggal', 'Observasi', 'Model_Raw', 'Terprediksi']].copy()
        df_final_export.rename(columns={'Model_Raw': 'ERA5 Sebelum Koreksi', 'Terprediksi': f'ERA5 Setelah Koreksi ({metode})'}, inplace=True)
        sub_text, f_name = f"Kolom: Tanggal · Observasi · ERA5 Sebelum Koreksi · ERA5 Setelah Koreksi ({metode})", f"Koreksi_Bias_{metode.replace(' ', '_')}.csv"

    col_dl1, col_dl2 = st.columns([3, 1])
    with col_dl1: st.markdown(f'<div style="padding:4px 0"><div class="dl-info-title">💾 Ekspor Hasil Analisis</div><div class="dl-info-sub">{sub_text}</div></div>', unsafe_allow_html=True)
    with col_dl2: st.download_button(label="📥 Unduh .CSV", data=df_final_export.to_csv(index=False).encode('utf-8'), file_name=f_name, mime="text/csv", use_container_width=True)
    with st.expander("👁️ Klik untuk melihat intipan tabel data"): st.dataframe(df_final_export.head(200), use_container_width=True, height=320)