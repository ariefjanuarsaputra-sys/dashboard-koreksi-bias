import streamlit as st
import pandas as pd
import numpy as np
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
# CSS PREMIUM - DESAIN PROFESIONAL
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
        font-size: 11px;
        background: #e8f2fc;
        color: #1a6bbf;
        border-radius: 20px;
        padding: 3px 10px;
        font-weight: 600;
    }
    .section-body {
        padding: 20px;
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

    /* Tombol bawaan Streamlit agar rapi */
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

    /* Hide default Streamlit metric styling */
    div[data-testid="stMetric"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# FUNGSI BANTU
# =========================================================================
def calculate_nse(obs, sim):
    pembagi = np.sum((obs - np.mean(obs)) ** 2)
    return np.nan if pembagi == 0 else 1 - (np.sum((obs - sim) ** 2) / pembagi)

def calculate_mbe(obs, sim):
    return np.mean(sim - obs)

def nse_badge(nse):
    if np.isnan(nse):
        return "badge-info", "N/A"
    if nse > 0.7:
        return "badge-good", "Sangat Baik"
    if nse > 0.4:
        return "badge-warn", "Cukup"
    return "badge-danger", "Perlu Perbaikan"

def mbe_badge(mbe):
    if abs(mbe) < 1:
        return "badge-good", "Minim Bias"
    if abs(mbe) < 3:
        return "badge-warn", f"{'Bias +' if mbe > 0 else 'Bias -'}"
    return "badge-danger", f"{'Bias +' if mbe > 0 else 'Bias -'}"


# =========================================================================
# LOAD DATA
# =========================================================================
@st.cache_data
def load_data():
    df_obs   = pd.read_excel("Minang_obs 1991-2025.xlsx")
    df_model = pd.read_excel("Book1.xlsx")
    df_obs.rename(columns={'RR': 'Observasi'}, inplace=True)
    df_model.rename(columns={'RR': 'Model_Raw'}, inplace=True)
    df_obs['Observasi']   = pd.to_numeric(df_obs['Observasi'],   errors='coerce')
    df_model['Model_Raw'] = pd.to_numeric(df_model['Model_Raw'], errors='coerce')
    df_obs['Tanggal']   = pd.to_datetime(df_obs['Tanggal'])
    df_model['Tanggal'] = pd.to_datetime(df_model['Tanggal'])
    return pd.merge(df_obs, df_model, on='Tanggal', how='inner')

try:
    df = load_data()
except Exception:
    st.error("⚠️ Gagal membaca data. Pastikan file Excel berada di folder yang sama dengan script ini.")
    st.stop()


# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-section">⚙️ Metode Koreksi</div>', unsafe_allow_html=True)
    metode = st.selectbox(
        "Pilih Metode:",
        ["Sebelum Koreksi (Model Raw)", "Delta Method", "Linear Scaling", "Variance Scaling", "Quantile Mapping"],
        label_visibility="collapsed"
    )

    faktor_bobot = 0.6
    if metode == "Delta Method":
        faktor_bobot = st.slider("Faktor Bobot Delta:", 0.0, 1.0, 0.6, 0.1)

    st.markdown('<div class="sidebar-section">📅 Rentang Waktu Grafik</div>', unsafe_allow_html=True)
    min_date = df['Tanggal'].min().date()
    max_date = df['Tanggal'].max().date()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        start_date = st.date_input("Dari", value=min_date, min_value=min_date, max_value=max_date, label_visibility="visible")
    with col_s2:
        end_date = st.date_input("Sampai", value=max_date, min_value=min_date, max_value=max_date, label_visibility="visible")

    st.markdown('<div class="sidebar-section">📍 Info Stasiun</div>', unsafe_allow_html=True)
    total_data   = len(df)
    missing_obs  = df['Observasi'].isna().sum()
    st.markdown(f"""
    <table class="info-table">
        <tr><td>Nama Stasiun</td><td>Minangkabau</td></tr>
        <tr><td>Kabupaten</td><td>Padang Pariaman</td></tr>
        <tr><td>Elevasi</td><td>3 m dpl</td></tr>
        <tr><td>Lintang</td><td>-0.7866°</td></tr>
        <tr><td>Bujur</td><td>100.2813°</td></tr>
        <tr><td>Total Data</td><td>{total_data:,} hari</td></tr>
        <tr><td>Data Hilang</td><td style="color:#b84820">{missing_obs} hari</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">💡 Panduan Nilai NSE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="guide-item guide-good">NSE &gt; 0.7 &nbsp;→&nbsp; Model sangat baik</div>
    <div class="guide-item guide-warn">NSE 0.4 – 0.7 &nbsp;→&nbsp; Cukup baik</div>
    <div class="guide-item guide-danger">NSE &lt; 0.4 &nbsp;→&nbsp; Perlu perbaikan</div>
    <div style="font-size:11px;color:#9ba3b4;margin-top:8px">
        MBE mendekati 0 = sedikit bias.<br>RMSE lebih kecil = lebih akurat.
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# HITUNG KOREKSI BIAS
# =========================================================================
mask_valid   = df['Observasi'].notna() & df['Model_Raw'].notna()
obs_clean    = df.loc[mask_valid, 'Observasi'].to_numpy().astype(float)
model_clean  = df.loc[mask_valid, 'Model_Raw'].to_numpy().astype(float)
waktu_clean  = df.loc[mask_valid, 'Tanggal'].values
df['Terprediksi'] = np.nan

if metode == "Sebelum Koreksi (Model Raw)":
    df['Terprediksi'] = df['Model_Raw']

elif metode == "Delta Method":
    if len(obs_clean) > 0:
        bulan_clean  = df.loc[mask_valid, 'Tanggal'].dt.month.to_numpy()
        hasil_delta  = model_clean.copy()
        for b in range(1, 13):
            mask_b = bulan_clean == b
            if np.sum(mask_b) > 0:
                selisih = (np.mean(obs_clean[mask_b]) - np.mean(model_clean[mask_b])) * faktor_bobot
                hasil_delta[mask_b] = model_clean[mask_b] + selisih
        df.loc[mask_valid, 'Terprediksi'] = np.clip(hasil_delta, 0.0, None)

else:
    try:
        import xarray as xr
        from cmethods import adjust
        obs_da   = xr.DataArray(obs_clean,   coords=[waktu_clean], dims=['time'], name='curah_hujan')
        model_da = xr.DataArray(model_clean, coords=[waktu_clean], dims=['time'], name='curah_hujan')
        if metode == "Linear Scaling":
            hasil_xr = adjust(method='linear_scaling',   obs=obs_da, simh=model_da, simp=model_da, kind='*')
        elif metode == "Variance Scaling":
            hasil_xr = adjust(method='variance_scaling', obs=obs_da, simh=model_da, simp=model_da, kind='+')
        elif metode == "Quantile Mapping":
            hasil_xr = adjust(method='quantile_mapping', obs=obs_da, simh=model_da, simp=model_da, n_quantiles=100, kind='*')
        df.loc[mask_valid, 'Terprediksi'] = np.clip(hasil_xr['curah_hujan'].values, 0.0, None)
    except Exception as err:
        st.sidebar.error(f"Gagal menjalankan cmethods: {err}")
        df['Terprediksi'] = df['Model_Raw']


# =========================================================================
# HITUNG METRIK
# =========================================================================
mask_metrik     = df['Observasi'].notna() & df['Terprediksi'].notna()
obs_metric      = df.loc[mask_metrik, 'Observasi']
sim_metric      = df.loc[mask_metrik, 'Terprediksi']

if len(obs_metric) > 0:
    mse  = mean_squared_error(obs_metric, sim_metric)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(obs_metric, sim_metric)
    mbe  = calculate_mbe(obs_metric, sim_metric)
    nse  = calculate_nse(obs_metric, sim_metric)
else:
    mse = rmse = mae = mbe = nse = 0.0

nse_cls, nse_lbl   = nse_badge(nse)
mbe_cls, mbe_lbl   = mbe_badge(mbe)
mbe_sign           = "+" if mbe > 0 else ""


# =========================================================================
# HEADER BANNER
# =========================================================================
st.markdown(f"""
<div class="header-banner">
    <div class="header-top-row">
        <p class="header-title">🌦️ Dashboard Analisis &amp; Koreksi Bias Curah Hujan</p>
        <span class="header-badge">✅ {metode}</span>
    </div>
    <div class="header-bottom-row">
        <span class="header-chip">📍 Stasiun Meteorologi Minangkabau, Kab. Padang Pariaman</span>
        <span class="header-chip-sep">·</span>
        <span class="header-chip">📅 1991 – 2025</span>
        <span class="header-chip-sep">·</span>
        <span class="header-chip">🗂️ {total_data:,} titik data</span>
        <span class="header-chip-sep">·</span>
        <span class="header-chip">⚠️ {missing_obs} data hilang</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# KARTU METRIK (HTML Custom — tidak terpotong)
# =========================================================================
st.markdown(f"""
<div class="metrics-row">
    <div class="metric-card blue">
        <div class="metric-label">MAE</div>
        <div class="metric-value blue">{mae:.3f}</div>
        <div class="metric-desc">Mean Absolute Error</div>
        <span class="metric-badge badge-info">mm/hari</span>
    </div>
    <div class="metric-card teal">
        <div class="metric-label">MSE</div>
        <div class="metric-value teal">{mse:.3f}</div>
        <div class="metric-desc">Mean Squared Error</div>
        <span class="metric-badge badge-info">mm²</span>
    </div>
    <div class="metric-card amber">
        <div class="metric-label">RMSE</div>
        <div class="metric-value amber">{rmse:.3f}</div>
        <div class="metric-desc">Root Mean Squared Error</div>
        <span class="metric-badge badge-warn">mm/hari</span>
    </div>
    <div class="metric-card coral">
        <div class="metric-label">MBE</div>
        <div class="metric-value coral">{mbe_sign}{mbe:.3f}</div>
        <div class="metric-desc">Mean Bias Error</div>
        <span class="metric-badge {mbe_cls}">{mbe_lbl}</span>
    </div>
    <div class="metric-card purple">
        <div class="metric-label">NSE</div>
        <div class="metric-value purple">{nse:.3f}</div>
        <div class="metric-desc">Nash-Sutcliffe Efficiency</div>
        <span class="metric-badge {nse_cls}">{nse_lbl}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# GRAFIK — 3 TAB
# =========================================================================
df_filtered = df[
    (df['Tanggal'].dt.date >= start_date) &
    (df['Tanggal'].dt.date <= end_date)
]
df['Bulan'] = df['Tanggal'].dt.month
monthly_avg = df.groupby('Bulan')[['Observasi', 'Model_Raw', 'Terprediksi']].mean().reset_index()
bulan_label = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Ags','Sep','Okt','Nov','Des']

st.markdown(f"""
<div class="section-header" style="background:#fff;border-radius:14px 14px 0 0;padding:14px 20px;border-bottom:1px solid #f0f2f5;display:flex;align-items:center;justify-content:space-between;box-shadow:0 2px 10px rgba(0,0,0,0.06);">
    <span class="section-title">📈 Visualisasi Perbandingan</span>
    <span class="method-pill">Metode: {metode}</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📉 Deret Waktu Harian", "📊 Klimatologi Bulanan", "🎯 Scatter Plot 1:1"])

# --- TAB 1: TIME SERIES ---
with tab1:
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_filtered['Tanggal'], y=df_filtered['Observasi'],
        mode='lines', name='Observasi BMKG',
        line=dict(color='#2980d4', width=1.5),
        connectgaps=False
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_filtered['Tanggal'], y=df_filtered['Terprediksi'],
        mode='lines', name=f'Terprediksi ({metode})',
        line=dict(color='#D85A30', width=1.5, dash='dot'),
        connectgaps=False
    ))
    fig_ts.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_title="Tanggal",
        yaxis_title="Curah Hujan (mm/hari)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0f2f5', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='#f0f2f5', gridwidth=1),
        font=dict(size=12)
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    n_hari = len(df_filtered)
    st.caption(f"Menampilkan {n_hari:,} hari data dari {start_date} hingga {end_date}.")

# --- TAB 2: KLIMATOLOGI BULANAN ---
with tab2:
    fig_clim = go.Figure()
    fig_clim.add_trace(go.Scatter(
        x=monthly_avg['Bulan'], y=monthly_avg['Observasi'],
        mode='lines+markers', name='Observasi BMKG',
        line=dict(color='#2980d4', width=3),
        marker=dict(size=8, symbol='circle')
    ))
    fig_clim.add_trace(go.Scatter(
        x=monthly_avg['Bulan'], y=monthly_avg['Model_Raw'],
        mode='lines+markers', name='Model Mentah',
        line=dict(color='#D62728', width=2, dash='dot'),
        marker=dict(size=6, symbol='x')
    ))
    fig_clim.add_trace(go.Scatter(
        x=monthly_avg['Bulan'], y=monthly_avg['Terprediksi'],
        mode='lines+markers', name=f'Setelah Koreksi ({metode})',
        line=dict(color='#1D9E75', width=2.5, dash='dash'),
        marker=dict(size=7, symbol='diamond')
    ))
    fig_clim.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=bulan_label,
            title="Bulan",
            showgrid=True, gridcolor='#f0f2f5'
        ),
        yaxis=dict(title="Rata-rata Curah Hujan (mm/hari)", showgrid=True, gridcolor='#f0f2f5'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    st.plotly_chart(fig_clim, use_container_width=True)
    st.caption("Rata-rata harian per bulan selama seluruh periode 1991–2025.")

# --- TAB 3: SCATTER PLOT ---
with tab3:
    if len(obs_metric) > 1:
        x_val = obs_metric.to_numpy().astype(float)
        y_val = sim_metric.to_numpy().astype(float)
        m, c  = np.polyfit(x_val, y_val, 1)
        r2    = np.corrcoef(x_val, y_val)[0, 1] ** 2
        pearson = np.corrcoef(x_val, y_val)[0, 1]
        tanda_c = "+" if c >= 0 else "-"

        # Banner regresi
        st.markdown(f"""
        <div class="reg-banner">
            📈 <strong>Hasil Analisis Regresi:</strong>
            &nbsp;&nbsp; Persamaan: <code>y = {m:.3f}x {tanda_c} {abs(c):.3f}</code>
            &nbsp;|&nbsp; R² = <strong>{r2:.3f}</strong>
            &nbsp;|&nbsp; Korelasi Pearson: <strong>{pearson:.3f}</strong>
        </div>
        """, unsafe_allow_html=True)

        max_val = max(x_val.max(), y_val.max()) * 1.05
        x_line  = np.linspace(0, max_val, 200)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=x_val, y=y_val, mode='markers',
            name='Data Harian',
            marker=dict(color='rgba(41, 128, 212, 0.30)', size=4, line=dict(width=0))
        ))
        fig_sc.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', name='Garis Ideal 1:1 (y = x)',
            line=dict(color='#D85A30', dash='dash', width=2)
        ))
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=m * x_line + c,
            mode='lines', name=f'Regresi: y = {m:.3f}x {tanda_c} {abs(c):.3f}',
            line=dict(color='#1D9E75', width=2.5)
        ))
        fig_sc.update_layout(
            xaxis_title="Curah Hujan Observasi BMKG (mm)",
            yaxis_title=f"Curah Hujan Terprediksi (mm)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=460,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f2f5', zeroline=True, zerolinecolor='#e0e0e0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f2f5', zeroline=True, zerolinecolor='#e0e0e0'),
            font=dict(size=12)
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.warning("Data tidak mencukupi untuk analisis regresi.")


# =========================================================================
# DOWNLOAD
# =========================================================================
st.markdown("---")
df_export = df.drop(columns=['Bulan'], errors='ignore')
csv_data  = df_export.to_csv(index=False).encode('utf-8')

col_dl1, col_dl2, col_dl3 = st.columns([3, 1, 1])
with col_dl1:
    st.markdown(f"""
    <div style="padding:4px 0">
        <div class="dl-info-title">💾 Ekspor Hasil Analisis</div>
        <div class="dl-info-sub">Kolom: Tanggal · Observasi · Model Raw · Terprediksi ({metode})</div>
    </div>
    """, unsafe_allow_html=True)
with col_dl2:
    st.download_button(
        label="📥 Unduh .CSV",
        data=csv_data,
        file_name=f"Koreksi_Bias_{metode.replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )
with col_dl3:
    with st.expander("👁️ Pratinjau Tabel"):
        pass

with st.expander("👁️ Klik untuk melihat intipan tabel data"):
    st.dataframe(df_export.head(200), use_container_width=True, height=320)
