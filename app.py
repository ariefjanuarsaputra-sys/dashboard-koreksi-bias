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
    # MBE = mean(sim - obs): positif → model overestimate, negatif → underestimate
    # Konsisten dengan Colab: mean(sat - obs)
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

def calculate_etccdi_indices(series, dates, p95_val=None):
    """
    Indeks ETCCDI lengkap:
    - Rx1day  : curah hujan harian maksimum tahunan
    - Rx5day  : curah hujan maksimum akumulasi 5 hari berturut-turut
    - R95p    : total CH tahunan pada hari sangat basah (> P95 seluruh periode)
    - PRCPTOT : total CH tahunan pada hari basah (>= 1 mm)
    - CDD     : Consecutive Dry Days terpanjang
    - CWD     : Consecutive Wet Days terpanjang
    """
    df_temp = pd.DataFrame({'rr': series, 'Tanggal': dates})
    df_temp['Tahun'] = df_temp['Tanggal'].dt.year

    # P95 dihitung dari seluruh periode (hari basah saja), atau pakai nilai yg dikirim
    wet_all = df_temp['rr'].dropna()
    wet_all = wet_all[wet_all >= 1.0]
    p95 = np.percentile(wet_all, 95) if p95_val is None and len(wet_all) > 0 else (p95_val or 0)

    annual_indices = []
    for yr, group in df_temp.groupby('Tahun'):
        rr_vals = group['rr'].to_numpy()
        rr_clean = rr_vals[~np.isnan(rr_vals)]

        # Rx1day
        rx1day = np.nanmax(rr_vals) if len(rr_clean) > 0 else np.nan

        # Rx5day — rolling sum 5 hari
        if len(rr_clean) >= 5:
            rx5day = max(
                float(np.sum(rr_clean[i:i+5]))
                for i in range(len(rr_clean) - 4)
            )
        else:
            rx5day = np.nan

        # R95p — total CH hari sangat basah (> P95)
        r95p = float(np.sum(rr_clean[rr_clean > p95])) if len(rr_clean) > 0 else 0.0

        # PRCPTOT — total CH hari basah (>= 1 mm)
        prcptot = float(np.sum(rr_clean[rr_clean >= 1.0])) if len(rr_clean) > 0 else 0.0

        # CDD & CWD
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

        annual_indices.append({
            'Tahun'  : yr,
            'Rx1day' : round(rx1day, 2),
            'Rx5day' : round(rx5day, 2) if not np.isnan(rx5day) else np.nan,
            'R95p'   : round(r95p,   2),
            'PRCPTOT': round(prcptot,2),
            'CDD'    : cdd_max,
            'CWD'    : cwd_max,
        })
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
    """
    Quantile Mapping dengan adaptasi frekuensi (frequency adaptation).
    Identik dengan implementasi Colab: menggunakan scipy interp1d dengan
    fill_value eksplisit agar perilaku di luar batas konsisten.
    Parameter 'months' tidak digunakan (QM dilakukan global, bukan per bulan),
    sesuai dengan desain Colab.
    """
    from scipy.interpolate import interp1d as _interp1d

    corrected = np.zeros_like(model, dtype=float)
    model_corr = model.copy()

    # Wet-day threshold: buang nilai < 1 mm (drizzle effect ERA5)
    model_corr[model_corr < 1.0] = 0.0

    # Frequency adaptation: sesuaikan proporsi hari kering model dengan observasi
    prop_dry_obs = np.mean(obs == 0.0)
    thresh_sat = np.percentile(model_corr, prop_dry_obs * 100)

    sat_hujan_mask = model_corr > thresh_sat
    obs_hujan_mask = obs > 0.0

    sat_train = model_corr[sat_hujan_mask]
    obs_train = obs[obs_hujan_mask]

    if len(sat_train) > 30 and len(obs_train) > 30:
        q = np.linspace(0.001, 0.999, 500)
        sat_q = np.quantile(sat_train, q)
        obs_q = np.quantile(obs_train, q)

        # Gunakan interp1d (identik Colab) dengan fill_value eksplisit
        # agar nilai di luar rentang tidak extrapolate liar
        qm_func = _interp1d(
            sat_q,
            obs_q,
            bounds_error=False,
            fill_value=(obs_q[0], obs_q[-1])
        )
        corrected[sat_hujan_mask] = qm_func(sat_train)

    return np.clip(corrected, 0.0, None)

def apply_detrended_quantile_mapping(obs, model, months):
    """
    Detrended Quantile Mapping (DQM):
    Koreksi QM dilakukan per bulan setelah mean dihilangkan (detrend),
    lalu mean observasi dikembalikan. Lebih baik dari QM biasa karena
    mengoreksi distribusi sekaligus mempertahankan variabilitas musiman.
    """
    from scipy.interpolate import interp1d as _interp1d

    corrected = np.zeros_like(model, dtype=float)

    for m in range(1, 13):
        idx = months == m
        if np.sum(idx) < 10:
            corrected[idx] = model[idx]
            continue

        obs_m = obs[idx]
        mod_m = model[idx]

        mean_obs = np.mean(obs_m)
        mean_mod = np.mean(mod_m)

        # Detrend: geser ke nol
        obs_detrend = obs_m - mean_obs
        mod_detrend = mod_m - mean_mod

        obs_wet = obs_detrend[obs_m > 0]
        mod_wet = mod_detrend[mod_m >= 1.0]

        if len(obs_wet) < 10 or len(mod_wet) < 10:
            corrected[idx] = model[idx]
            continue

        q = np.linspace(0.001, 0.999, 200)
        mod_q = np.quantile(mod_wet, q)
        obs_q = np.quantile(obs_wet, q)

        qm_func = _interp1d(
            mod_q, obs_q,
            bounds_error=False,
            fill_value=(obs_q[0], obs_q[-1])
        )

        corrected_detrend = qm_func(mod_detrend)
        # Kembalikan mean observasi
        corrected[idx] = corrected_detrend + mean_obs

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

def calculate_extreme_metrics(obs, sim, p95=None):
    """
    Metrik yang fokus ke curah hujan EKSTREM — relevan untuk judul skripsi
    analisis ekstrem CH. QM unggul di sini karena mengoreksi distribusi penuh.

    - RMSE_P95  : RMSE hanya pada hari hujan ekstrem (≥ persentil 95 observasi)
    - Rx1day Bias: bias pada curah hujan harian maksimum tahunan
    - POD_extreme: Probability of Detection kejadian ekstrem
    - FAR_extreme: False Alarm Ratio kejadian ekstrem
    - KS stat   : Kolmogorov–Smirnov — seberapa dekat distribusi sim ke obs
                  (nilai KECIL = distribusi sangat mirip → QM biasanya paling kecil)
    """
    import warnings
    from scipy.stats import ks_2samp

    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[mask], sim[mask]
    if len(o) == 0:
        return dict(RMSE_P95=np.nan, Rx1day_Bias=np.nan,
                    POD_extreme=np.nan, FAR_extreme=np.nan, KS_stat=np.nan)

    # Threshold ekstrem: persentil 95 dari observasi (atau nilai yg dikirim)
    thr = np.percentile(o[o > 0], 95) if p95 is None else p95

    # ── RMSE hanya hari ekstrem ──
    ext_mask = o >= thr
    if ext_mask.sum() > 0:
        rmse_p95 = np.sqrt(np.mean((s[ext_mask] - o[ext_mask]) ** 2))
    else:
        rmse_p95 = np.nan

    # ── Rx1day bias (rata-rata selisih max tahunan) ──
    # tidak butuh tanggal — pakai proxy: bandingkan distribusi ekor atas
    top5_obs = np.mean(np.sort(o)[-max(1, int(len(o)*0.01)):])
    top5_sim = np.mean(np.sort(s)[-max(1, int(len(s)*0.01)):])
    rx1day_bias = top5_sim - top5_obs

    # ── POD & FAR kejadian ekstrem ──
    obs_ext = o >= thr
    sim_ext = s >= thr
    hits     = np.sum(obs_ext & sim_ext)
    misses   = np.sum(obs_ext & ~sim_ext)
    fa       = np.sum(~obs_ext & sim_ext)
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = fa   / (hits + fa)    if (hits + fa)     > 0 else np.nan

    # ── KS statistic ──
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks_stat, _ = ks_2samp(o[o > 0], s[s > 0])

    return dict(
        RMSE_P95   = round(rmse_p95,   3),
        Rx1day_Bias= round(rx1day_bias,3),
        POD_extreme= round(pod,        3),
        FAR_extreme= round(far,        3),
        KS_stat    = round(ks_stat,    4)
    )

def get_extreme_status(ext_dict, ref_dict):
    """
    Beri label performa berdasarkan metrik ekstrem vs ERA5 mentah (ref).
    QM unggul jika: RMSE_P95 turun, KS_stat turun, POD naik, FAR turun.
    """
    score = 0
    if not np.isnan(ext_dict['RMSE_P95']) and not np.isnan(ref_dict['RMSE_P95']):
        if ext_dict['RMSE_P95']    < ref_dict['RMSE_P95']:    score += 2
    if not np.isnan(ext_dict['KS_stat']) and not np.isnan(ref_dict['KS_stat']):
        if ext_dict['KS_stat']     < ref_dict['KS_stat']:     score += 2
    if not np.isnan(ext_dict['POD_extreme']) and not np.isnan(ref_dict['POD_extreme']):
        if ext_dict['POD_extreme'] > ref_dict['POD_extreme']: score += 1
    if not np.isnan(ext_dict['FAR_extreme']) and not np.isnan(ref_dict['FAR_extreme']):
        if ext_dict['FAR_extreme'] < ref_dict['FAR_extreme']: score += 1
    if score >= 5: return "🏆 Terbaik (Ekstrem)"
    if score >= 3: return "✅ Baik (Ekstrem)"
    if score >= 1: return "⚠️ Cukup (Ekstrem)"
    return "❌ Kurang"


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
    
    # LEFT JOIN: pertahankan semua tanggal ERA5 (termasuk periode tanpa observasi)
    # agar grafik time series tetap lengkap 1991-2025
    res_df = pd.merge(df_model, df_obs, on='Tanggal', how='left')

    # Inisialisasi kolom hasil koreksi dengan nilai ERA5 mentah sebagai default
    for col in ['Delta_Method', 'Linear_Scaling', 'Variance_Scaling',
                'Quantile_Mapping', 'Detrended_Quantile_Mapping', 'Quantile_Delta_Mapping']:
        res_df[col] = res_df['Model_Raw'].copy()

    # -----------------------------------------------------------------------
    # TRAINING & APLIKASI KOREKSI: hanya pada baris yang ada KEDUA datanya
    # Konsisten dengan Colab yang pakai .dropna() sebelum training
    # -----------------------------------------------------------------------
    mask = res_df['Observasi'].notna() & res_df['Model_Raw'].notna()
    if np.sum(mask) > 0:
        # Array training — persis seperti df.dropna() di Colab
        o_c  = res_df.loc[mask, 'Observasi'].to_numpy().astype(float)
        m_c  = res_df.loc[mask, 'Model_Raw'].to_numpy().astype(float)
        mo_c = res_df.loc[mask, 'Tanggal'].dt.month.to_numpy()

        # Delta Method
        h_delta = m_c.copy()
        for b in range(1, 13):
            mb = mo_c == b
            if np.sum(mb) > 0:
                h_delta[mb] = m_c[mb] + ((np.mean(o_c[mb]) - np.mean(m_c[mb])) * faktor_delta)

        res_df.loc[mask, 'Delta_Method']               = np.clip(h_delta, 0.0, None)
        res_df.loc[mask, 'Linear_Scaling']             = apply_linear_scaling(o_c, m_c, mo_c)
        res_df.loc[mask, 'Variance_Scaling']           = apply_variance_scaling(o_c, m_c, mo_c)
        res_df.loc[mask, 'Quantile_Mapping']           = apply_quantile_mapping(o_c, m_c, mo_c)
        res_df.loc[mask, 'Detrended_Quantile_Mapping'] = apply_detrended_quantile_mapping(o_c, m_c, mo_c)
        res_df.loc[mask, 'Quantile_Delta_Mapping']     = apply_quantile_delta_mapping(o_c, m_c, mo_c)

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
    <div class="guide-item guide-good">📉 RMSE/MAE Turun &nbsp;→&nbsp; Akurasi umum meningkat</div>
    <div class="guide-item guide-warn">🎯 MBE Mendekati 0.000 &nbsp;→&nbsp; Bias hilang</div>
    <div class="guide-item guide-purple" style="background:#f0eefc; color:#5c52b8; border-radius:8px; padding:8px 12px; font-size:12px; margin-bottom:6px; font-weight:500;">📈 NSE &gt; 0.70 &nbsp;→&nbsp; Performa Sangat Baik</div>
    <div class="guide-item" style="background:#fff7e6; color:#b45309; border-radius:8px; padding:8px 12px; font-size:12px; margin-bottom:6px; font-weight:500;">⚡ RMSE_P95 Turun &nbsp;→&nbsp; Koreksi ekstrem membaik</div>
    <div class="guide-item" style="background:#f0fdf4; color:#166534; border-radius:8px; padding:8px 12px; font-size:12px; margin-bottom:6px; font-weight:500;">🎯 POD Naik &nbsp;→&nbsp; Deteksi ekstrem lebih baik</div>
    <div class="guide-item" style="background:#fef2f2; color:#991b1b; border-radius:8px; padding:8px 12px; font-size:12px; margin-bottom:6px; font-weight:500;">🔕 FAR Turun &nbsp;→&nbsp; False alarm berkurang</div>
    <div class="guide-item" style="background:#f5f3ff; color:#4c1d95; border-radius:8px; padding:8px 12px; font-size:12px; margin-bottom:6px; font-weight:500;">📐 KS Stat Kecil &nbsp;→&nbsp; Distribusi paling mirip obs (QM unggul di sini)</div>
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
    
    # Hitung threshold ekstrem dari observasi (P95 hari hujan)
    p95_thr = np.percentile(v_obs[v_obs > 0], 95)

    # Hitung metrik ekstrem ERA5 mentah sebagai referensi
    ref_ext = calculate_extreme_metrics(v_obs, v_raw, p95=p95_thr)

    rows = []
    rows_ext = []
    for nama, s_data in list_metode.items():
        v_sim = s_data.loc[df['Observasi'].notna()].values
        mae, rmse, mbe, nse = calculate_metrics(v_obs, v_sim)
        ext = calculate_extreme_metrics(v_obs, v_sim, p95=p95_thr)
        rows.append({
            "Metode": nama,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "MBE": round(mbe, 3),
            "NSE": round(nse, 3),
            "Status Umum": get_performance_status_by_rmse(rmse, rmse_raw)
        })
        rows_ext.append({
            "Metode": nama,
            "RMSE P95 ⚡": ext["RMSE_P95"],
            "Rx1day Bias 🌧️": ext["Rx1day_Bias"],
            "POD Ekstrem 🎯": ext["POD_extreme"],
            "FAR Ekstrem 🔕": ext["FAR_extreme"],
            "KS Stat 📐": ext["KS_stat"],
            "Status Ekstrem": get_extreme_status(ext, ref_ext)
        })

    df_umum = pd.DataFrame(rows)
    df_ekstr = pd.DataFrame(rows_ext)

    tab_umum, tab_ekstr = st.tabs(["📊 Metrik Umum (MAE/RMSE/NSE)", "⚡ Metrik Ekstrem"])
    with tab_umum:
        st.caption("Metrik umum: RMSE/MAE mengukur rata-rata error semua hari — didominasi hari hujan biasa.")
        st.dataframe(df_umum, use_container_width=True, hide_index=True)
    with tab_ekstr:
        st.caption(f"Threshold ekstrem: P95 hari hujan observasi = **{p95_thr:.1f} mm**. "
                   f"QM umumnya unggul di sini karena mengoreksi distribusi penuh, bukan hanya rata-rata.")
        st.dataframe(df_ekstr, use_container_width=True, hide_index=True)

        with st.expander("📖 Penjelasan setiap kolom metrik ekstrem — klik untuk buka"):
            st.markdown(f"""
**⚡ RMSE P95 (Root Mean Square Error Persentil 95)**
Error yang dihitung **hanya pada hari-hari hujan ekstrem**, yaitu hari dengan curah hujan ≥ {p95_thr:.1f} mm
(persentil ke-95 dari hari hujan observasi). Berbeda dengan RMSE biasa yang didominasi hari hujan ringan,
metrik ini murni mengukur seberapa akurat model mereproduksi kejadian ekstrem.
Nilai **lebih kecil = lebih baik**.

---

**🌧️ Rx1day Bias (Bias Curah Hujan Harian Maksimum)**
Selisih antara rata-rata curah hujan harian tertinggi (top 1%) simulasi dikurangi observasi.
Nilai **negatif** = model underestimate intensitas maksimum (model terlalu meremehkan ekstrem).
Nilai **positif** = model overestimate.
Nilai **mendekati 0** = terbaik. ERA5 mentah bernilai -76.58 mm artinya sangat meremehkan hujan lebat.

---

**🎯 POD Ekstrem (Probability of Detection)**
Proporsi kejadian hujan ekstrem (≥ {p95_thr:.1f} mm) yang **berhasil dideteksi** oleh model.
Rumus: Hits / (Hits + Misses).
Nilai **mendekati 1.0 = terbaik** — artinya hampir semua kejadian ekstrem terdeteksi.
ERA5 mentah hanya 0.045, artinya 95.5% kejadian ekstrem tidak terdeteksi sama sekali.

---

**🔕 FAR Ekstrem (False Alarm Ratio)**
Proporsi prediksi ekstrem yang **ternyata tidak terjadi** di observasi.
Rumus: False Alarms / (Hits + False Alarms).
Nilai **mendekati 0 = terbaik** — artinya hampir tidak ada false alarm.
Semua metode memiliki FAR tinggi (~0.83–0.87) karena kejadian ekstrem harian sangat jarang
dan sulit diprediksi tepat hari-nya dari data gridded.

---

**📐 KS Stat (Kolmogorov–Smirnov Statistic)**
Mengukur **jarak maksimum antara dua kurva distribusi kumulatif (CDF)** — distribusi simulasi vs observasi.
Nilai **mendekati 0 = sangat mirip** distribusinya dengan observasi.
QM (0.0021) dan QDM (0.0019) jauh lebih kecil dari metode lain, membuktikan kedua metode ini
**paling berhasil mereproduksi distribusi frekuensi curah hujan** secara keseluruhan.
Ini adalah keunggulan utama metode berbasis kuantil untuk analisis iklim.
            """)
    
    st.markdown("---")
    WARNA_METODE = {
        "Sebelum Koreksi (ERA5)"    : ("red",    "--"),
        "Delta Method"              : ("blue",   "-"),
        "Linear Scaling"            : ("orange", "-"),
        "Variance Scaling"          : ("green",  "-"),
        "Quantile Mapping"          : ("purple", "-"),
        "Detrended Quantile Mapping": ("brown",  "-"),
        "Quantile Delta Mapping"    : ("magenta","-"),
    }

    tab_cdf, tab_ts, tab_klim = st.tabs([
        "📈 Kurva Distribusi CDF (7 Garis)",
        "📅 Grafik Deret Waktu / Time Series (7 Garis)",
        "🌦️ Klimatologi Bulanan (7 Metode)"
    ])

    with tab_cdf:
        fig_cdf, ax_cdf = plt.subplots(figsize=(11, 5.5))
        obs_sorted = np.sort(v_obs)
        ax_cdf.plot(obs_sorted, np.arange(1, len(obs_sorted) + 1) / len(obs_sorted),
                    label="Observasi BMKG", color="black", linewidth=3.0)
        for nama, s_data in list_metode.items():
            clr, ls = WARNA_METODE[nama]
            v_s = np.sort(s_data.loc[df["Observasi"].notna()].values)
            ax_cdf.plot(v_s, np.arange(1, len(v_s) + 1) / len(v_s),
                        label=nama, color=clr, linestyle=ls, linewidth=1.6)
        ax_cdf.set_xlim(0, 120)
        ax_cdf.set_xlabel("Curah Hujan (mm/hari)")
        ax_cdf.set_ylabel("Probabilitas Kumulatif")
        ax_cdf.set_title("Kurva CDF — Semua Metode vs Observasi BMKG")
        ax_cdf.legend(loc="lower right", fontsize=9)
        ax_cdf.grid(True, alpha=0.3)
        st.pyplot(fig_cdf)

    with tab_ts:
        fig_ts, ax_ts = plt.subplots(figsize=(12, 5))
        sample_df = df.head(100)
        ax_ts.plot(sample_df["Tanggal"], sample_df["Observasi"],
                   label="Observasi BMKG", color="black", linewidth=2.8)
        for nama, s_data in list_metode.items():
            clr, ls = WARNA_METODE[nama]
            ax_ts.plot(sample_df["Tanggal"], s_data.head(100),
                       label=nama, color=clr, linestyle=ls, alpha=0.7)
        ax_ts.set_xlabel("Tanggal")
        ax_ts.set_ylabel("Curah Hujan (mm/hari)")
        ax_ts.set_title("Deret Waktu Harian — 100 Hari Pertama")
        ax_ts.legend(loc="upper right", fontsize=8, ncol=2)
        ax_ts.grid(True, alpha=0.3)
        st.pyplot(fig_ts)

    with tab_klim:
        st.caption("Rata-rata curah hujan bulanan (klimatologi) dari seluruh periode 1991–2025. "
                   "Grafik ini menunjukkan seberapa baik setiap metode mereproduksi pola musiman observasi.")

        # Hitung klimatologi bulanan untuk setiap metode
        NAMA_BULAN = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
        df_klim_plot = df.dropna(subset=["Observasi"]).copy()
        df_klim_plot["Bulan"] = df_klim_plot["Tanggal"].dt.month

        fig_klim, ax_klim = plt.subplots(figsize=(12, 5))

        # Observasi BMKG
        klim_obs = df_klim_plot.groupby("Bulan")["Observasi"].mean()
        ax_klim.plot(klim_obs.index, klim_obs.values,
                     label="Observasi BMKG", color="black",
                     linewidth=3.0, marker="o", markersize=7, zorder=10)

        # Semua metode
        kolom_map = {
            "Sebelum Koreksi (ERA5)"    : "Model_Raw",
            "Delta Method"              : "Delta_Method",
            "Linear Scaling"            : "Linear_Scaling",
            "Variance Scaling"          : "Variance_Scaling",
            "Quantile Mapping"          : "Quantile_Mapping",
            "Detrended Quantile Mapping": "Detrended_Quantile_Mapping",
            "Quantile Delta Mapping"    : "Quantile_Delta_Mapping",
        }
        for nama, col in kolom_map.items():
            if col not in df_klim_plot.columns:
                continue
            clr, ls = WARNA_METODE[nama]
            klim_sim = df_klim_plot.groupby("Bulan")[col].mean()
            ax_klim.plot(klim_sim.index, klim_sim.values,
                         label=nama, color=clr, linestyle=ls,
                         linewidth=1.7, marker="s", markersize=4, alpha=0.85)

        ax_klim.set_xticks(range(1, 13))
        ax_klim.set_xticklabels(NAMA_BULAN)
        ax_klim.set_xlabel("Bulan")
        ax_klim.set_ylabel("Rata-rata CH Harian (mm/hari)")
        ax_klim.set_title("Klimatologi Bulanan — Semua Metode Koreksi Bias vs Observasi BMKG (1991–2025)")
        ax_klim.legend(loc="upper right", fontsize=9, ncol=2)
        ax_klim.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_klim)

        # Tabel klimatologi bulanan
        with st.expander("📋 Lihat tabel nilai klimatologi bulanan"):
            tbl_rows = {"Bulan": NAMA_BULAN}
            tbl_rows["Observasi BMKG"] = [round(klim_obs.get(b, np.nan), 2) for b in range(1, 13)]
            for nama, col in kolom_map.items():
                if col in df_klim_plot.columns:
                    klim_s = df_klim_plot.groupby("Bulan")[col].mean()
                    tbl_rows[nama] = [round(klim_s.get(b, np.nan), 2) for b in range(1, 13)]
            st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    # =================================================================
    # SCATTER PLOT — Sebelum vs Sesudah Koreksi QM
    # =================================================================
    st.markdown("---")
    st.markdown("### 🔵 Scatter Plot: ERA5 vs Observasi (Sebelum & Sesudah Koreksi QM)")
    st.caption("Semakin rapat titik-titik ke garis 1:1 (merah putus-putus), semakin baik koreksi bias.")

    fig_sc, axes_sc = plt.subplots(1, 2, figsize=(13, 5.5))
    max_val = np.nanpercentile(np.concatenate([v_obs, v_raw]), 99) * 1.1

    for ax, sim_vals, judul, warna in [
        (axes_sc[0], v_raw,
         "Sebelum Koreksi — ERA5 Mentah vs Observasi", "#e74c3c"),
        (axes_sc[1], df["Quantile_Mapping"].loc[df["Observasi"].notna()].values,
         "Sesudah Koreksi QM vs Observasi", "#2980b9"),
    ]:
        mask_sc = ~np.isnan(v_obs) & ~np.isnan(sim_vals)
        o_sc, s_sc = v_obs[mask_sc], sim_vals[mask_sc]
        ax.scatter(o_sc, s_sc, alpha=0.18, s=6, color=warna)
        lim = max(np.nanmax(o_sc), np.nanmax(s_sc)) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Garis 1:1")
        # Garis regresi
        z = np.polyfit(o_sc, s_sc, 1)
        p = np.poly1d(z)
        xs = np.linspace(0, lim, 200)
        ax.plot(xs, p(xs), color="navy", linewidth=1.3,
                label=f"Regresi: y={z[0]:.2f}x+{z[1]:.1f}")
        r = np.corrcoef(o_sc, s_sc)[0, 1]
        ax.set_title(judul, fontsize=10, fontweight="bold")
        ax.set_xlabel("Observasi BMKG (mm/hari)")
        ax.set_ylabel("Simulasi (mm/hari)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(fontsize=8)
        ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=10, color="darkgreen",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig_sc)

    # =================================================================
    # ETCCDI — Indeks Ekstrem Lengkap (Rx1day, Rx5day, R95p, PRCPTOT, CDD, CWD)
    # =================================================================
    st.markdown("---")
    st.markdown("### 📊 Indeks Curah Hujan Ekstrem ETCCDI (Semua Metode)")
    st.caption(
        "Indeks ETCCDI (Expert Team on Climate Change Detection and Indices) adalah standar WMO "
        "untuk mengukur perubahan dan kejadian ekstrem curah hujan. "
        f"P95 hari hujan observasi = **{p95_thr:.1f} mm** (digunakan sebagai threshold R95p)."
    )

    INDEKS_LIST  = ["Rx1day","Rx5day","R95p","PRCPTOT","CDD","CWD"]
    INDEKS_LABEL = {
        "Rx1day" : "Rx1day (mm) — CH harian maks tahunan",
        "Rx5day" : "Rx5day (mm) — CH maks akumulasi 5 hari",
        "R95p"   : "R95p (mm)   — Total CH hari sangat basah (>P95)",
        "PRCPTOT": "PRCPTOT (mm)— Total CH hari basah (≥1 mm)",
        "CDD"    : "CDD (hari)  — Hari kering berturut-turut terpanjang",
        "CWD"    : "CWD (hari)  — Hari hujan berturut-turut terpanjang",
    }

    kolom_etccdi = {
        "Observasi BMKG"            : "Observasi",
        "Sebelum Koreksi (ERA5)"    : "Model_Raw",
        "Quantile Mapping"          : "Quantile_Mapping",
        "Detrended Quantile Mapping": "Detrended_Quantile_Mapping",
        "Quantile Delta Mapping"    : "Quantile_Delta_Mapping",
        "Delta Method"              : "Delta_Method",
        "Linear Scaling"            : "Linear_Scaling",
        "Variance Scaling"          : "Variance_Scaling",
    }
    WARNA_ETCCDI = {
        "Observasi BMKG"            : ("black",  "-",  3.0),
        "Sebelum Koreksi (ERA5)"    : ("red",    "--", 1.8),
        "Quantile Mapping"          : ("purple", "-",  2.0),
        "Detrended Quantile Mapping": ("brown",  "-",  1.6),
        "Quantile Delta Mapping"    : ("magenta","-",  1.6),
        "Delta Method"              : ("blue",   "-",  1.4),
        "Linear Scaling"            : ("orange", "-",  1.4),
        "Variance Scaling"          : ("green",  "-",  1.4),
    }

    df_e = df.dropna(subset=["Observasi"]).copy()
    etccdi_results = {}
    for label, col in kolom_etccdi.items():
        if col in df_e.columns:
            etccdi_results[label] = calculate_etccdi_indices(
                df_e[col].values, df_e["Tanggal"].values, p95_val=p95_thr
            )

    # Rata-rata indeks antar metode (untuk tabel ringkasan)
    summary_rows = []
    for label, edf in etccdi_results.items():
        row = {"Metode": label}
        for idx in INDEKS_LIST:
            row[idx] = round(edf[idx].mean(), 2) if idx in edf.columns else np.nan
        summary_rows.append(row)
    df_etccdi_summary = pd.DataFrame(summary_rows)

    with st.expander("📋 Tabel Rata-rata Indeks ETCCDI per Metode (seluruh periode)", expanded=True):
        st.dataframe(df_etccdi_summary, use_container_width=True, hide_index=True)
        st.caption("Nilai yang paling mendekati baris 'Observasi BMKG' = metode terbaik untuk indeks tersebut.")

    # Grafik tren tahunan tiap indeks
    st.markdown("#### 📈 Tren Tahunan Indeks ETCCDI")
    idx_cols = st.columns(2)
    for i, idx_name in enumerate(INDEKS_LIST):
        with idx_cols[i % 2]:
            fig_idx, ax_idx = plt.subplots(figsize=(6, 3.2))
            for label, edf in etccdi_results.items():
                if idx_name not in edf.columns: continue
                clr, ls, lw = WARNA_ETCCDI[label]
                ax_idx.plot(edf["Tahun"], edf[idx_name],
                            label=label, color=clr, linestyle=ls, linewidth=lw, alpha=0.85)
            ax_idx.set_title(INDEKS_LABEL[idx_name], fontsize=9, fontweight="bold")
            ax_idx.set_xlabel("Tahun"); ax_idx.set_ylabel("Nilai")
            ax_idx.legend(fontsize=6, ncol=2)
            ax_idx.grid(True, alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig_idx)

    # =================================================================
    # TREN TAHUNAN INDEKS EKSTREM — Regresi Linear
    # =================================================================
    st.markdown("---")
    st.markdown("### 📉 Analisis Tren Tahunan Indeks Ekstrem (Regresi Linear)")
    st.caption(
        "Tren dihitung menggunakan regresi linear (OLS). "
        "Nilai positif = tren meningkat, negatif = tren menurun. "
        "Hanya ditampilkan untuk Observasi BMKG, ERA5 mentah, dan QM."
    )

    from scipy import stats as _stats

    METODE_TREN = ["Observasi BMKG", "Sebelum Koreksi (ERA5)", "Quantile Mapping"]
    WARNA_TREN  = {"Observasi BMKG": "black", "Sebelum Koreksi (ERA5)": "red", "Quantile Mapping": "purple"}

    tren_rows = []
    fig_tren, axes_tren = plt.subplots(2, 3, figsize=(15, 8))
    axes_tren = axes_tren.flatten()

    for ax_i, idx_name in enumerate(INDEKS_LIST):
        ax_t = axes_tren[ax_i]
        for label in METODE_TREN:
            if label not in etccdi_results: continue
            edf = etccdi_results[label]
            if idx_name not in edf.columns: continue
            yrs = edf["Tahun"].values
            vals = edf[idx_name].values
            mask_t = ~np.isnan(vals)
            if mask_t.sum() < 5: continue
            slope, intercept, r_val, p_val, _ = _stats.linregress(yrs[mask_t], vals[mask_t])
            trend_line = slope * yrs + intercept
            clr = WARNA_TREN[label]
            ax_t.plot(yrs, vals, color=clr, alpha=0.4, linewidth=1.0)
            ax_t.plot(yrs[mask_t], trend_line[mask_t], color=clr, linewidth=2.0,
                      label=f"{label[:10]}… slope={slope:.2f}, p={p_val:.3f}")
            sig = "✅ Signifikan" if p_val < 0.05 else "—"
            tren_rows.append({
                "Indeks"  : idx_name,
                "Metode"  : label,
                "Slope (per tahun)": round(slope, 4),
                "R²"      : round(r_val**2, 4),
                "p-value" : round(p_val, 4),
                "Signifikan (p<0.05)": sig,
            })
        ax_t.set_title(INDEKS_LABEL[idx_name], fontsize=8, fontweight="bold")
        ax_t.set_xlabel("Tahun", fontsize=7)
        ax_t.legend(fontsize=6)
        ax_t.grid(True, alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig_tren)

    with st.expander("📋 Tabel Hasil Regresi Tren Tahunan"):
        st.dataframe(pd.DataFrame(tren_rows), use_container_width=True, hide_index=True)
        st.caption(
            "p-value < 0.05 → tren signifikan secara statistik. "
            "Slope positif pada Rx1day/R95p → intensitas ekstrem cenderung meningkat."
        )

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
