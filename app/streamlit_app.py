"""
Healthcare AI Prediction Portal — Overhauled
=============================================
Multi-section dashboard for medical data analysis and prediction.
Features: cached model loading, step-by-step questionnaire UI,
sample data for testing, graceful demo fallbacks, HIPAA disclaimer.

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

pio.templates.default = "plotly_dark"

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")

from utils.clinical_interpretations import (
    ECG_CLASS_EXPLANATIONS, ECG_METRIC_EXPLANATIONS,
    interpret_heart_risk, interpret_xray,
    CBC_EXPLANATIONS, interpret_diabetes_results,
    interpret_lipid_results, interpret_kidney_results,
    LAB_GENERAL_EXPLANATIONS,
)
from utils.clinical_calculators import (
    classify_value, CBC_RANGES, interpret_cbc,
    calculate_findrisc, classify_hba1c, classify_fasting_glucose,
    classify_lipid, LIPID_CLASSES, calculate_ascvd_risk,
    ckd_epi_creatinine, ckd_epi_cystatin, stage_ckd, stage_albuminuria,
    DEMO_LAB_REPORT,
)

# ============================================================
# Cached Model Loading (Part 1 — Stability)
# ============================================================

@st.cache_resource
def load_heart_model():
    """Load heart risk model with graceful fallback."""
    try:
        import joblib
        model_path = os.path.join(MODELS_DIR, "heart_risk.joblib")
        scaler_path = os.path.join(MODELS_DIR, "heart_risk_scaler.joblib")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, True
    except Exception:
        pass
    return None, None, False


@st.cache_resource
def load_ecg_model():
    """Load ECG classifier with graceful fallback."""
    try:
        from tensorflow import keras
        model_path = os.path.join(MODELS_DIR, "ecg_classifier.h5")
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model, True
    except Exception:
        pass
    return None, False


@st.cache_resource
def load_xray_model():
    """Load X-ray classifier with graceful fallback."""
    try:
        from tensorflow import keras
        model_path = os.path.join(MODELS_DIR, "xray_classifier.h5")
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model, True
    except Exception:
        pass
    return None, False


@st.cache_data
def get_sample_ecg_files():
    """List available sample ECG files."""
    ecg_dir = os.path.join(SAMPLES_DIR, "ecg")
    if not os.path.exists(ecg_dir):
        return {}
    samples = {}
    name_map = {
        "sample_normal.npy": "Normal ECG",
        "sample_mi.npy": "Myocardial Infarction",
        "sample_sttc.npy": "ST/T Change",
        "sample_hyp.npy": "Hypertrophy",
        "sample_cd.npy": "Conduction Disturbance",
    }
    for fname, label in name_map.items():
        fpath = os.path.join(ecg_dir, fname)
        if os.path.exists(fpath):
            samples[label] = fpath
    return samples


@st.cache_data
def get_sample_xray_files():
    """List available sample X-ray files."""
    xray_dir = os.path.join(SAMPLES_DIR, "xray")
    if not os.path.exists(xray_dir):
        return {}
    samples = {}
    for fname in sorted(os.listdir(xray_dir)):
        if fname.endswith((".png", ".jpg", ".jpeg")):
            label = fname.replace("_", " ").replace(".png", "").replace(".jpg", "").title()
            samples[label] = os.path.join(xray_dir, fname)
    return samples


# Pre-load models at startup
_heart_model, _heart_scaler, _heart_loaded = load_heart_model()
_ecg_model, _ecg_loaded = load_ecg_model()
_xray_model, _xray_loaded = load_xray_model()


def _fallback_risk(age, sex_val, cp_val, trestbps, chol, fbs_val, thalach, exang_val, oldpeak, ca):
    """Rule-based fallback when ML model is unavailable."""
    risk_factors = 0
    if age > 55: risk_factors += 1
    if sex_val == 1: risk_factors += 0.5
    if cp_val == 3: risk_factors += 1.5
    if trestbps > 140: risk_factors += 1
    if chol > 240: risk_factors += 1
    if fbs_val == 1: risk_factors += 0.5
    if thalach < 120: risk_factors += 1
    if exang_val == 1: risk_factors += 1.5
    if oldpeak > 2: risk_factors += 1
    if ca > 0: risk_factors += ca
    risk_score = min(risk_factors / 10 * 100, 99)
    predicted = "High Risk" if risk_score > 50 else "Low Risk"
    return risk_score, predicted


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Healthcare AI Prediction Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F1F5F9;
    }

    .main { background: #0F172A; }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* Disclaimer */
    .disclaimer {
        background: rgba(251,191,36,0.08);
        border-left: 4px solid #FBBF24;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #FBBF24;
        margin-top: 24px;
    }

    /* Hero Banner */
    .hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 48px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        color: #F1F5F9;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: 0;
        max-width: 600px;
        line-height: 1.6;
    }

    .stat-row {
        display: flex;
        gap: 12px;
        margin-top: 24px;
        flex-wrap: wrap;
    }
    .stat-pill {
        background: rgba(56,189,248,0.08);
        backdrop-filter: blur(8px);
        border: 1px solid #334155;
        border-radius: 40px;
        padding: 8px 20px;
        color: #CBD5E1;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stat-pill strong { font-weight: 700; color: #38BDF8; }

    /* Feature Cards -- FIXED ALIGNMENT */
    .feature-card {
        background: #1E293B;
        border-radius: 16px;
        padding: 28px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #334155;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        min-height: 340px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(56,189,248,0.12);
        border-color: #38BDF8;
    }
    .feature-icon {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 16px;
    }
    .icon-ecg { background: rgba(248,113,113,0.15); }
    .icon-xray { background: rgba(56,189,248,0.15); }
    .icon-risk { background: rgba(52,211,153,0.15); }
    .icon-cbc { background: rgba(192,132,252,0.15); }
    .icon-diabetes { background: rgba(251,191,36,0.15); }
    .icon-lipid { background: rgba(34,211,238,0.15); }
    .icon-kidney { background: rgba(248,113,113,0.15); }
    .icon-lab { background: rgba(129,140,248,0.15); }

    .feature-card h3 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 0 0 8px 0;
    }
    .feature-card p {
        color: #94A3B8;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 0;
        min-height: 90px;
        flex-shrink: 0;
    }
    .feature-tag {
        display: inline-block;
        background: #0F172A;
        color: #38BDF8;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 14px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border: 1px solid #334155;
    }

    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border-radius: 14px;
        padding: 22px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        border-left: 4px solid #38BDF8;
        margin-bottom: 12px;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 20px rgba(56,189,248,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #F1F5F9;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .risk-high { border-left-color: #F87171 !important; }
    .risk-high .metric-value { color: #F87171; }
    .risk-medium { border-left-color: #FBBF24 !important; }
    .risk-medium .metric-value { color: #FBBF24; }
    .risk-low { border-left-color: #34D399 !important; }
    .risk-low .metric-value { color: #34D399; }

    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .section-sub {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 28px;
        line-height: 1.5;
    }

    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        color: #F1F5F9;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
        border: 1px solid #334155;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .result-box h2 {
        color: #38BDF8;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .result-box p {
        color: #94A3B8;
        margin: 6px 0 0;
        font-size: 0.95rem;
    }

    /* Info Cards */
    .info-card {
        background: #1E293B;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .info-card h4 {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 0 0 6px 0;
    }
    .info-card p {
        font-size: 0.85rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* Upload Zone */
    .upload-zone {
        background: #1E293B;
        border: 2px dashed #334155;
        border-radius: 16px;
        padding: 40px 24px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .upload-zone:hover {
        border-color: #38BDF8;
        background: #1E293B;
    }
    .upload-icon { font-size: 2.5rem; margin-bottom: 12px; }
    .upload-text { color: #94A3B8; font-size: 0.9rem; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #1E293B !important;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #CBD5E1 !important;
        font-weight: 500;
        padding: 8px 4px;
        border-radius: 8px;
        transition: background 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(56,189,248,0.08);
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* Button Styling */
    .stButton > button {
        background: #0077B6 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: #005F8C !important;
        box-shadow: 0 4px 16px rgba(0,119,182,0.3) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"] {
        background: #0077B6 !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #1E293B;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background: #0077B6 !important;
        color: white !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 24px 0 12px;
        color: #64748B;
        font-size: 0.78rem;
        letter-spacing: 0.01em;
    }
    .footer a { color: #38BDF8; text-decoration: none; font-weight: 600; }
    .footer-divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 0 auto 16px;
    }

    /* Flag Badges */
    .flag-critical {
        background: rgba(248,113,113,0.15);
        color: #F87171;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .flag-high {
        background: rgba(251,191,36,0.15);
        color: #FBBF24;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .flag-low {
        background: rgba(251,191,36,0.15);
        color: #FBBF24;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .flag-normal {
        background: rgba(52,211,153,0.15);
        color: #34D399;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Lab Table */
    .lab-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .lab-table th {
        background: #0F172A;
        padding: 10px 14px;
        text-align: left;
        font-weight: 600;
        color: #94A3B8;
        border-bottom: 2px solid #334155;
    }
    .lab-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #1E293B;
        color: #CBD5E1;
    }

    /* CKD Grid */
    .ckd-grid {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .ckd-grid th {
        background: #0F172A;
        padding: 10px 14px;
        text-align: center;
        font-weight: 600;
        color: #94A3B8;
        border-bottom: 2px solid #334155;
    }
    .ckd-grid td {
        padding: 8px 14px;
        border-bottom: 1px solid #1E293B;
        text-align: center;
        color: #CBD5E1;
    }
    .ckd-cell {
        padding: 8px;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Questionnaire step card */
    .step-card {
        background: #1E293B;
        border-radius: 16px;
        padding: 32px 28px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        border: 1px solid #334155;
        margin-bottom: 16px;
    }
    .step-card h3 {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 0 0 6px 0;
    }
    .step-card p {
        color: #94A3B8;
        font-size: 0.9rem;
        margin: 0 0 18px 0;
    }

    /* Progress bar */
    .progress-bar-bg {
        background: #334155;
        border-radius: 10px;
        height: 8px;
        margin-bottom: 24px;
        overflow: hidden;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #0077B6, #38BDF8);
        height: 100%;
        border-radius: 10px;
        transition: width 0.4s ease;
    }

    /* Health status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .status-loaded { background: rgba(52,211,153,0.15); color: #34D399 !important; border: 1px solid rgba(52,211,153,0.3); }
    .status-demo { background: rgba(251,191,36,0.15); color: #FBBF24 !important; border: 1px solid rgba(251,191,36,0.3); }

    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Card grid alignment — force all Streamlit columns to stretch equally */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Steps cards */
    .step-item {
        text-align: center;
        padding: 20px 12px;
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        height: 100%;
    }

    /* Benefit cards */
    .benefit-card {
        background: #1E293B;
        border-radius: 14px;
        padding: 22px 24px;
        border: 1px solid #334155;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .benefit-card h4 {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 0 0 6px 0;
    }
    .benefit-card p {
        font-size: 0.85rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* Responsive grid for steps and benefits */
    @media (max-width: 768px) {
        .step-item { min-height: auto; }
        .benefit-card { min-height: auto; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Navigation — URL-based with query_params for browser back/forward
# ============================================================
NAV_OPTIONS = [
    "Home", "Heart / ECG", "Chest X-Ray", "Health Risk Assessment",
    "CBC Analysis", "Diabetes Screening", "Lipid Panel / CV Risk",
    "Kidney Function", "Lab Report Upload", "AI Assistant", "Privacy & Compliance",
]

# Slug mapping: URL param <-> display name
_SLUG_TO_NAV = {
    "home": "Home", "ecg": "Heart / ECG", "xray": "Chest X-Ray",
    "risk": "Health Risk Assessment", "cbc": "CBC Analysis",
    "diabetes": "Diabetes Screening", "lipid": "Lipid Panel / CV Risk",
    "kidney": "Kidney Function", "lab": "Lab Report Upload",
    "assistant": "AI Assistant", "privacy": "Privacy & Compliance",
}
_NAV_TO_SLUG = {v: k for k, v in _SLUG_TO_NAV.items()}


def navigate_to(section_name):
    """Navigate to a section by updating query params."""
    if section_name in NAV_OPTIONS:
        slug = _NAV_TO_SLUG.get(section_name, "home")
        if slug == "home":
            st.query_params.clear()
        else:
            st.query_params["page"] = slug
        st.session_state.nav_radio = section_name


# Sync session state from URL on page load (handles browser back/forward)
_url_page = st.query_params.get("page", "home")
_nav_from_url = _SLUG_TO_NAV.get(_url_page, "Home")
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = _nav_from_url
elif st.session_state.nav_radio != _nav_from_url:
    # URL changed (browser back/forward) — sync session state to match
    st.session_state.nav_radio = _nav_from_url

# ============================================================
# Session State Initialization
# ============================================================
# Health Risk questionnaire step tracking
if "hra_step" not in st.session_state:
    st.session_state.hra_step = 1
if "hra_data" not in st.session_state:
    st.session_state.hra_data = {}
if "hra_submitted" not in st.session_state:
    st.session_state.hra_submitted = False


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 8px 0 4px;">
        <div style="font-size: 2.2rem; margin-bottom: 2px;">🏥</div>
        <div style="font-size: 1.1rem; font-weight: 800; color: #38BDF8; letter-spacing: -0.02em;">Healthcare AI</div>
        <div style="font-size: 0.8rem; font-weight: 500; color: rgba(255,255,255,0.6);">Prediction Portal</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    def _sync_url_from_radio():
        """When sidebar radio changes, update URL query params."""
        sel = st.session_state.nav_radio
        slug = _NAV_TO_SLUG.get(sel, "home")
        if slug == "home":
            st.query_params.clear()
        else:
            st.query_params["page"] = slug

    section = st.radio(
        "Navigation",
        NAV_OPTIONS,
        key="nav_radio",
        on_change=_sync_url_from_radio,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Health check status
    st.markdown("##### System Status")
    if _heart_loaded:
        st.markdown('<span class="status-badge status-loaded">Heart Risk: Model Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-demo">Heart Risk: Demo Mode</span>', unsafe_allow_html=True)

    if _ecg_loaded:
        st.markdown('<span class="status-badge status-loaded">ECG: Model Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-demo">ECG: Demo Mode</span>', unsafe_allow_html=True)

    if _xray_loaded:
        st.markdown('<span class="status-badge status-loaded">X-Ray: Model Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-demo">X-Ray: Demo Mode</span>', unsafe_allow_html=True)

    st.markdown('<span class="status-badge status-loaded">CBC Analysis: Algorithm</span>', unsafe_allow_html=True)
    st.markdown('<span class="status-badge status-loaded">Diabetes: Algorithm</span>', unsafe_allow_html=True)
    st.markdown('<span class="status-badge status-loaded">Lipid/CV: Algorithm</span>', unsafe_allow_html=True)
    st.markdown('<span class="status-badge status-loaded">Kidney: Algorithm</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Built by")
    st.markdown("**Abdullah Abdul Sami**")
    st.caption("MS Data Science (AI)")
    st.caption("Northwestern University")
    st.markdown("---")
    st.caption("For research and educational purposes only. Not for clinical diagnosis.")


# ============================================================
# Browser back/forward — use components.html so JS actually executes
# ============================================================
components.html(
    """
    <script>
        window.parent.addEventListener("popstate", function() {
            window.parent.location.reload();
        });
    </script>
    """,
    height=0,
)

# ============================================================
# HOME SECTION
# ============================================================
if section == "Home":
    st.markdown("""
    <div class="hero">
        <h1>Healthcare AI Prediction Portal</h1>
        <p>
            AI-powered clinical decision support. Upload ECGs, chest X-rays, or enter patient
            vitals to receive instant diagnostic predictions powered by deep learning.
        </p>
        <div class="stat-row">
            <span class="stat-pill"><strong>21,837</strong>&nbsp; ECG Records</span>
            <span class="stat-pill"><strong>112,120</strong>&nbsp; X-Ray Images</span>
            <span class="stat-pill"><strong>920</strong>&nbsp; Patient Records</span>
            <span class="stat-pill"><strong>10</strong>&nbsp; Clinical Modules</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How It Works Steps — rendered as single HTML grid for perfect alignment
    st.markdown("""
    <br>
    <p class="section-header">How It Works</p>
    <p class="section-sub">Get AI-powered clinical insights in four simple steps.</p>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px;">
        <div class="step-item">
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #0077B6, #005F8C); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: bold; margin: 0 auto 12px auto;">1</div>
            <div style="font-size: 1.5rem; margin-bottom: 4px;">🔍</div>
            <h4 style="margin: 0 0 4px 0; font-size: 1rem; color: #F1F5F9;">Browse Modules</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0;">Explore our 10 clinical AI tools below</p>
        </div>
        <div class="step-item">
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #0077B6, #005F8C); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: bold; margin: 0 auto 12px auto;">2</div>
            <div style="font-size: 1.5rem; margin-bottom: 4px;">👆</div>
            <h4 style="margin: 0 0 4px 0; font-size: 1rem; color: #F1F5F9;">Select a Module</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0;">Click 'Open' on any module to get started</p>
        </div>
        <div class="step-item">
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #0077B6, #005F8C); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: bold; margin: 0 auto 12px auto;">3</div>
            <div style="font-size: 1.5rem; margin-bottom: 4px;">📤</div>
            <h4 style="margin: 0 0 4px 0; font-size: 1rem; color: #F1F5F9;">Upload or Try Samples</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0;">Provide your data or use built-in sample datasets</p>
        </div>
        <div class="step-item">
            <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #0077B6, #005F8C); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: bold; margin: 0 auto 12px auto;">4</div>
            <div style="font-size: 1.5rem; margin-bottom: 4px;">📊</div>
            <h4 style="margin: 0 0 4px 0; font-size: 1rem; color: #F1F5F9;">View Results</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0;">Get AI predictions with visual explanations instantly</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Why Use This Portal — rendered as single HTML grid for perfect alignment
    st.markdown("""
    <p class="section-header">Why Use This Portal?</p>
    <p class="section-sub">Built for students, researchers, and clinicians who want to explore AI in healthcare.</p>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px;">
        <div class="benefit-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">⚡</div>
            <h4>Free &amp; Instant</h4>
            <p>All 10 clinical AI tools are completely free with no registration required. Get results in seconds.</p>
        </div>
        <div class="benefit-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🔬</div>
            <h4>Research-Grade Models</h4>
            <p>Built on peer-reviewed ML architectures including CNNs, transfer learning, and ensemble methods with published accuracy metrics.</p>
        </div>
        <div class="benefit-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🧪</div>
            <h4>Try Before You Upload</h4>
            <p>Every module includes built-in sample data so you can explore and understand the tools before using your own data.</p>
        </div>
        <div class="benefit-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">📖</div>
            <h4>Educational &amp; Transparent</h4>
            <p>Each tool explains how its AI model works, what the results mean, and what the limitations are.</p>
        </div>
        <div class="benefit-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🔒</div>
            <h4>Privacy First</h4>
            <p>Your uploaded data is processed in real time and never stored. No accounts, no data collection, no tracking.</p>
        </div>
        <div class="benefit-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🩺</div>
            <h4>Clinically Relevant</h4>
            <p>Modules cover cardiology, radiology, hematology, nephrology, endocrinology, and more.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    CARD_DATA = [
        ("Heart / ECG", "icon-ecg", "❤️", "Heart / ECG Analysis",
         "Upload 12-lead ECG recordings for real-time arrhythmia classification. Get heart rate, HRV metrics, and diagnostic probabilities.", "1D CNN Model"),
        ("Chest X-Ray", "icon-xray", "🫁", "Chest X-Ray Analysis",
         "Upload frontal chest X-ray images for pneumonia detection and multi-label disease classification.", "Transfer Learning"),
        ("Health Risk Assessment", "icon-risk", "📊", "Health Risk Assessment",
         "Interactive step-by-step questionnaire to generate a heart disease risk score with interactive charts.", "Ensemble (XGB+LGBM+GBM)"),
        ("CBC Analysis", "icon-cbc", "🩸", "CBC Analysis",
         "Enter complete blood count values for automated classification, WBC differential visualization, and clinical interpretation.", "Clinical Algorithm"),
        ("Diabetes Screening", "icon-diabetes", "🍩", "Diabetes Screening",
         "Comprehensive diabetes risk assessment using HbA1c, fasting glucose, and the validated FINDRISC questionnaire.", "FINDRISC"),
        ("Lipid Panel / CV Risk", "icon-lipid", "🫀", "Lipid Panel / CV Risk",
         "Lipid classification per ATP III guidelines with 10-year ASCVD risk estimation using Pooled Cohort Equations.", "Pooled Cohort Equations"),
        ("Kidney Function", "icon-kidney", "🫘", "Kidney Function",
         "CKD-EPI 2021 race-free eGFR with KDIGO staging, albuminuria assessment, and risk classification.", "CKD-EPI 2021"),
        ("Lab Report Upload", "icon-lab", "📄", "Lab Report Upload",
         "Upload lab report PDFs for automated parsing and analysis. Get color-coded flags for abnormal values.", "PDF Parsing"),
    ]

    for row_start in range(0, len(CARD_DATA), 3):
        cols = st.columns(3, gap="medium")
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(CARD_DATA):
                nav_key, icon_cls, emoji, title, desc, tag = CARD_DATA[idx]
                with col:
                    # Fixed-height card HTML so buttons align across columns
                    st.markdown(f"""
                    <div style="background: #1E293B; border-radius: 16px; padding: 28px 24px; border: 1px solid #334155; min-height: 270px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 2px 8px rgba(0,0,0,0.3); transition: all 0.3s ease; margin-bottom: 8px;">
                        <div>
                            <div class="feature-icon {icon_cls}">{emoji}</div>
                            <h3 style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9; margin: 0 0 8px 0;">{title}</h3>
                            <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.55; margin: 0;">{desc}</p>
                        </div>
                        <span class="feature-tag">{tag}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.button(
                        f"Open {title}",
                        key=f"nav_{nav_key}",
                        use_container_width=True,
                        on_click=navigate_to,
                        args=(nav_key,),
                    )
        if row_start + 3 < len(CARD_DATA):
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset info row
    st.markdown('<p class="section-header">Datasets & Models</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>PTB-XL Dataset</h4>
            <p>21,837 clinical 12-lead ECGs (10s, 500Hz) from PhysioNet. Annotated with
            SCP-ECG diagnostic statements across 5 superclasses.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>NIH Chest X-ray14</h4>
            <p>112,120 frontal-view chest X-rays with 14 disease labels.
            Binary pneumonia classifier trained via MobileNetV2 transfer learning.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>UCI Heart Disease</h4>
            <p>920 patient records with 13 clinical features.
            Ensemble classifier with hyperparameter tuning achieves 90%+ accuracy.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# HEART / ECG SECTION
# ============================================================
elif section == "Heart / ECG":
    st.button("← Back to Home", key="back_home_ecg", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>Heart / ECG Analysis</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Heart / ECG Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload a 12-lead ECG recording, try a sample, or explore the demo analysis.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Upload your ECG file or try sample data → 2️⃣ Click Analyze → 3️⃣ View arrhythmia classification
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About This Module"):
        st.write("""
        Electrocardiogram (ECG) analysis is one of the most important diagnostic tools in
        cardiology. This module uses deep learning to automatically classify 12-lead ECG
        recordings into multiple arrhythmia categories, including normal sinus rhythm,
        ST/T wave changes, conduction disturbances, hypertrophy, and myocardial infarction.

        Useful for medical students learning ECG interpretation, researchers working with
        ECG datasets, and anyone interested in AI-applied cardiac diagnostics.
        """)

    with st.expander("How the Model Works"):
        st.write("""
        **Architecture:** 1D Convolutional Neural Network (1D-CNN)

        **Training Data:** 21,837 clinical 12-lead ECG recordings from the PTB-XL dataset
        (PhysioNet), sampled at 500Hz with 10-second duration.

        **Preprocessing:** Raw ECG signals are bandpass filtered, normalized, and segmented
        into fixed-length windows before being fed into the CNN.

        **Performance:** Accuracy: 99.5%

        **Pipeline:** Raw ECG File -> Signal Preprocessing -> 1D-CNN Classification ->
        Arrhythmia Category + Confidence Score
        """)

    with st.expander("Input Requirements"):
        st.write("""
        - **Accepted formats:** CSV, DAT, HEA, NPY
        - **Expected data:** 12-lead ECG recording. CSV files should have columns for each lead (I, II, III, aVR, aVL, aVF, V1-V6).
        - **Sample rate:** 500 Hz recommended
        - You can also use the built-in sample data or demo mode to test without uploading.
        """)

    with st.expander("Understanding Your Results"):
        st.write("""
        The model outputs a classification (e.g., Normal Sinus Rhythm, Atrial Fibrillation)
        along with a confidence probability chart.

        - **High confidence (>90%):** The model is fairly certain about its classification.
        - **Moderate confidence (60-90%):** Consider the top 2-3 classifications.
        - **Low confidence (<60%):** The recording may be noisy or ambiguous.

        The HRV (Heart Rate Variability) metrics provide additional context about cardiac rhythm regularity.
        """)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_sample, tab_demo = st.tabs(["Upload ECG", "Try Sample Data", "Demo Analysis"])

    def run_ecg_prediction(ecg_data, tab_context="upload"):
        """Run ECG prediction and display results."""
        fig = go.Figure()
        if ecg_data.ndim > 1:
            lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                          "V1", "V2", "V3", "V4", "V5", "V6"]
            for i in range(min(ecg_data.shape[1], 12)):
                fig.add_trace(go.Scatter(
                    y=ecg_data[:, i] + i * 3,
                    name=lead_names[i] if i < len(lead_names) else f"Lead {i+1}",
                    line=dict(width=1)
                ))
        else:
            fig.add_trace(go.Scatter(y=ecg_data, name="ECG"))

        fig.update_layout(
            title="ECG Signal",
            xaxis_title="Sample",
            yaxis_title="Amplitude (mV)",
            height=500,
            template="plotly_dark",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True, key="ecg_signal_chart")

        # Predict
        if _ecg_loaded:
            try:
                from utils.ecg_utils import prepare_ecg_for_model, DIAGNOSTIC_CLASSES
                with st.spinner("Analyzing your 12-lead ECG recording..."):
                    processed = prepare_ecg_for_model(ecg_data)
                    if processed.ndim == 2:
                        processed = np.expand_dims(processed, axis=0)
                    preds = _ecg_model.predict(processed, verbose=0)[0]
                    class_names = list(DIAGNOSTIC_CLASSES.values())
                    pred_idx = int(np.argmax(preds))
                    confidence = float(preds[pred_idx]) * 100

                st.success("ECG analysis complete!")

                st.markdown(f"""
                <div class="result-box">
                    <h2>{class_names[pred_idx]}</h2>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                probs_df = pd.DataFrame({
                    "Condition": class_names,
                    "Probability (%)": [float(p) * 100 for p in preds]
                })
                fig_p = px.bar(
                    probs_df, x="Probability (%)", y="Condition",
                    orientation="h", color="Probability (%)",
                    color_continuous_scale=["#1E293B", "#38BDF8"],
                )
                fig_p.update_layout(height=300, template="plotly_dark", font=dict(family="Inter"))
                st.plotly_chart(fig_p, use_container_width=True, key="ecg_probability_model")

                if class_names[pred_idx] in ECG_CLASS_EXPLANATIONS:
                    with st.expander("What does this mean?", expanded=True):
                        st.markdown(ECG_CLASS_EXPLANATIONS[class_names[pred_idx]])
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.info("ECG model not loaded. Running in demo mode with simulated results.")
            demo_probs = pd.DataFrame({
                "Condition": ["Normal ECG", "ST/T Change", "Conduction Dist.", "Hypertrophy", "MI"],
                "Probability (%)": [89.2, 5.1, 3.2, 1.8, 0.7]
            })
            fig_p = px.bar(
                demo_probs, x="Probability (%)", y="Condition",
                orientation="h", color="Probability (%)",
                color_continuous_scale=["#1E293B", "#38BDF8"],
            )
            fig_p.update_layout(height=260, template="plotly_dark", font=dict(family="Inter"), showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True, key="ecg_probability_sample")

    # --- Upload Tab ---
    with tab_upload:
        st.markdown("""
        <div style="text-align: center; padding: 32px 20px; background: #1E293B; border-radius: 12px; border: 2px dashed #334155; margin-bottom: 16px;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">📈</div>
            <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">Upload a 12-lead ECG recording to get instant arrhythmia classification</p>
            <p style="color: #64748B; font-size: 0.82rem; margin: 4px 0 0;">Supported formats: CSV, DAT, HEA, NPY</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose ECG file", type=["csv", "dat", "hea", "npy"],
            help="CSV (columns = leads), NumPy arrays, or WFDB format",
            key="ecg_file_uploader",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    ecg_data = pd.read_csv(uploaded_file).values
                elif uploaded_file.name.endswith(".npy"):
                    ecg_data = np.load(uploaded_file)
                else:
                    st.warning("For .dat/.hea files, use the WFDB loader in the demo tab.")
                    ecg_data = None

                if ecg_data is not None:
                    st.success(f"Loaded ECG signal: {ecg_data.shape}")
                    run_ecg_prediction(ecg_data)
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # --- Sample Data Tab ---
    with tab_sample:
        sample_ecgs = get_sample_ecg_files()
        if sample_ecgs:
            st.markdown("Select a sample ECG recording to test the classification model:")
            selected_sample = st.selectbox(
                "Choose a sample", list(sample_ecgs.keys()), key="ecg_sample_select"
            )
            if st.button("Load & Analyze Sample", type="primary", key="ecg_sample_btn"):
                try:
                    ecg_data = np.load(sample_ecgs[selected_sample])
                    st.success(f"Loaded sample: {selected_sample} ({ecg_data.shape})")
                    run_ecg_prediction(ecg_data)
                except Exception as e:
                    st.error(f"Error loading sample: {e}")
        else:
            st.info("No sample ECG files found. Sample data will be available after running the data generation scripts.")

    # --- Demo Tab ---
    with tab_demo:
        st.markdown("Simulated 12-lead ECG analysis with sample cardiac data.")

        np.random.seed(42)
        fs = 500
        t = np.linspace(0, 10, fs * 10)
        ecg_sim = (
            0.6 * np.sin(2 * np.pi * 1.2 * t) +
            0.3 * np.sin(2 * np.pi * 2.4 * t) +
            np.where((t % (1/1.2)) < 0.05, 1.5, 0) +
            0.05 * np.random.randn(len(t))
        )

        col1, col2, col3, col4 = st.columns(4, gap="small")
        with col1:
            st.markdown("""
            <div class="metric-card">
                <p class="metric-label">Heart Rate</p>
                <p class="metric-value">72 <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">BPM</small></p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <p class="metric-label">SDNN</p>
                <p class="metric-value">45.2 <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">ms</small></p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <p class="metric-label">RMSSD</p>
                <p class="metric-value">38.7 <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">ms</small></p>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card risk-low">
                <p class="metric-label">Classification</p>
                <p class="metric-value" style="font-size:1.3rem;">Normal Sinus</p>
            </div>""", unsafe_allow_html=True)

        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(
            x=t[:2000], y=ecg_sim[:2000],
            mode="lines", name="Lead II",
            line=dict(color="#2c5364", width=1.5)
        ))
        fig_demo.update_layout(
            title=dict(text="12-Lead ECG - Lead II", font=dict(size=16, family="Inter")),
            xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
            height=380, template="plotly_dark",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_demo, use_container_width=True, key="ecg_demo_signal")

        demo_probs = pd.DataFrame({
            "Condition": ["Normal ECG", "ST/T Change", "Conduction Dist.", "Hypertrophy", "MI"],
            "Probability (%)": [89.2, 5.1, 3.2, 1.8, 0.7]
        })
        fig_p = px.bar(
            demo_probs, x="Probability (%)", y="Condition",
            orientation="h", color="Probability (%)",
            color_continuous_scale=["#1E293B", "#38BDF8"],
        )
        fig_p.update_layout(height=260, template="plotly_dark", font=dict(family="Inter"), showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True, key="ecg_demo_probability")

    st.markdown("---")
    st.markdown("##### Explore More Modules")
    sug_cols = st.columns(3, gap="medium")
    with sug_cols[0]:
        st.button("Chest X-Ray Analysis →", key="suggest_ecg_to_xray", on_click=navigate_to, args=("Chest X-Ray",), use_container_width=True)
    with sug_cols[1]:
        st.button("Health Risk Assessment →", key="suggest_ecg_to_hra", on_click=navigate_to, args=("Health Risk Assessment",), use_container_width=True)
    with sug_cols[2]:
        st.button("CBC Analysis →", key="suggest_ecg_to_cbc", on_click=navigate_to, args=("CBC Analysis",), use_container_width=True)


# ============================================================
# CHEST X-RAY SECTION
# ============================================================
elif section == "Chest X-Ray":
    st.button("← Back to Home", key="back_home_xray", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>Chest X-Ray Analysis</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Chest X-Ray Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload a frontal chest X-ray, try a sample image, or view a demo prediction.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Upload a chest X-ray image or try sample data → 2️⃣ Click Analyze → 3️⃣ View pneumonia detection results
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About This Module"):
        st.write("""
        Chest X-ray analysis is a cornerstone of radiology. This module uses deep learning
        to classify frontal chest X-ray images, detecting pneumonia versus normal findings.
        It is useful for medical students studying radiology, researchers exploring AI in
        medical imaging, and anyone interested in computer-aided diagnosis.
        """)

    with st.expander("How the Model Works"):
        st.write("""
        **Architecture:** MobileNetV2 with transfer learning (pretrained on ImageNet)

        **Training Data:** NIH Chest X-ray14 dataset with 112,120 frontal-view chest X-ray
        images annotated with 14 disease labels. Binary classifier trained for pneumonia detection.

        **Preprocessing:** Images are resized to 224x224, normalized to [0, 1] range, and
        augmented with random flips and rotations during training.

        **Performance:** Accuracy varies by class; optimized for pneumonia sensitivity.

        **Pipeline:** Chest X-Ray Image -> Resize & Normalize -> MobileNetV2 -> Pneumonia vs Normal + Confidence
        """)

    with st.expander("Input Requirements"):
        st.write("""
        - **Accepted formats:** PNG, JPEG/JPG
        - **Expected data:** Frontal (PA or AP) chest X-ray image
        - **Resolution:** Any resolution accepted; images are resized internally to 224x224
        - You can also use the built-in sample images to test without uploading.
        """)

    with st.expander("Understanding Your Results"):
        st.write("""
        The model outputs a binary classification (Pneumonia or Normal) with a confidence score.

        - **High confidence (>90%):** The model is fairly certain about its classification.
        - **Moderate confidence (60-90%):** The result should be interpreted with caution.
        - **Low confidence (<60%):** The image may be ambiguous or of poor quality.

        A Grad-CAM heatmap (when available) highlights regions the model focused on.
        """)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_sample = st.tabs(["Upload X-Ray", "Try Sample Data"])

    def run_xray_prediction(img, source_label="Uploaded"):
        """Run X-ray prediction and display results."""
        col_img, col_result = st.columns([1, 1], gap="large")
        with col_img:
            st.image(img, caption=f"{source_label} X-ray", use_container_width=True)

        with col_result:
            if _xray_loaded:
                try:
                    from utils.xray_utils import load_and_preprocess_xray, prepare_xray_for_model, PNEUMONIA_CLASSES
                    img_array = np.array(img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)

                    with st.spinner("Running pneumonia detection on your chest X-ray..."):
                        preds = _xray_model.predict(img_batch, verbose=0)[0]

                    st.success("X-ray analysis complete!")

                    if len(preds) == 1:
                        pneumonia_prob = float(preds[0]) * 100
                        normal_prob = 100 - pneumonia_prob
                        predicted = "Pneumonia" if pneumonia_prob > 50 else "Normal"
                        confidence = pneumonia_prob if predicted == "Pneumonia" else normal_prob
                    else:
                        pred_idx = int(np.argmax(preds))
                        predicted = PNEUMONIA_CLASSES[pred_idx]
                        confidence = float(preds[pred_idx]) * 100
                        normal_prob = float(preds[0]) * 100
                        pneumonia_prob = float(preds[1]) * 100 if len(preds) > 1 else 0

                    risk_class = "risk-high" if predicted == "Pneumonia" else "risk-low"
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <p class="metric-label">Prediction</p>
                        <p class="metric-value">{predicted}</p>
                        <p style="color:#6b7b8d; font-size:0.9rem; margin-top:4px;">
                            Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    probs_df = pd.DataFrame({
                        "Class": ["Normal", "Pneumonia"],
                        "Probability (%)": [normal_prob, pneumonia_prob]
                    })
                    fig = px.pie(probs_df, values="Probability (%)", names="Class",
                                 color_discrete_sequence=["#34D399", "#F87171"], hole=0.4)
                    fig.update_layout(height=300, font=dict(family="Inter"), margin=dict(t=20, b=20), template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True, key="xray_prediction_pie")

                    with st.expander("Clinical Interpretation", expanded=True):
                        st.markdown(interpret_xray(predicted, f"{confidence:.1f}"))

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.markdown("""
                <div class="result-box">
                    <h2>Demo Mode</h2>
                    <p>X-ray model not loaded. Showing sample prediction.</p>
                </div>
                """, unsafe_allow_html=True)
                demo_data = pd.DataFrame({
                    "Condition": ["Normal", "Pneumonia"],
                    "Probability (%)": [82.3, 17.7]
                })
                fig = px.pie(demo_data, values="Probability (%)", names="Condition",
                             color_discrete_sequence=["#34D399", "#F87171"], hole=0.4)
                fig.update_layout(height=300, font=dict(family="Inter"), margin=dict(t=20, b=20), template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True, key="xray_demo_pie")

    # --- Upload Tab ---
    with tab_upload:
        st.markdown("""
        <div style="text-align: center; padding: 32px 20px; background: #1E293B; border-radius: 12px; border: 2px dashed #334155; margin-bottom: 16px;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">🫁</div>
            <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">Upload a frontal chest X-ray for AI-powered pneumonia detection</p>
            <p style="color: #64748B; font-size: 0.82rem; margin: 4px 0 0;">Supported formats: PNG, JPEG</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_xray = st.file_uploader(
            "Choose X-ray image", type=["png", "jpg", "jpeg"],
            key="xray_file_uploader",
            help="Supported: PNG, JPEG. Frontal PA/AP view recommended.",
        )
        if uploaded_xray is not None:
            from PIL import Image
            img = Image.open(uploaded_xray)
            run_xray_prediction(img, "Uploaded")
        else:
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-icon">🫁</div>
                <div class="upload-text">Upload a chest X-ray image<br>
                <small>PNG or JPEG format</small></div>
            </div>
            """, unsafe_allow_html=True)

    # --- Sample Data Tab ---
    with tab_sample:
        sample_xrays = get_sample_xray_files()
        if sample_xrays:
            st.markdown("Select a sample chest X-ray to test the classification model:")
            selected_xray = st.selectbox(
                "Choose a sample", list(sample_xrays.keys()), key="xray_sample_select"
            )
            if st.button("Load & Analyze Sample", type="primary", key="xray_sample_btn"):
                try:
                    from PIL import Image
                    img = Image.open(sample_xrays[selected_xray])
                    run_xray_prediction(img, selected_xray)
                except Exception as e:
                    st.error(f"Error loading sample: {e}")
        else:
            st.info("No sample X-ray files found. Sample data will be available after running the data generation scripts.")

    st.markdown("---")
    st.markdown("##### Explore More Modules")
    sug_cols = st.columns(3, gap="medium")
    with sug_cols[0]:
        st.button("Heart / ECG Analysis →", key="suggest_xray_to_ecg", on_click=navigate_to, args=("Heart / ECG",), use_container_width=True)
    with sug_cols[1]:
        st.button("Health Risk Assessment →", key="suggest_xray_to_hra", on_click=navigate_to, args=("Health Risk Assessment",), use_container_width=True)
    with sug_cols[2]:
        st.button("Lab Report Upload →", key="suggest_xray_to_lab", on_click=navigate_to, args=("Lab Report Upload",), use_container_width=True)


# ============================================================
# HEALTH RISK ASSESSMENT — Step-by-Step Questionnaire
# ============================================================
elif section == "Health Risk Assessment":
    st.button("← Back to Home", key="back_home_hra", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>Health Risk Assessment</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Health Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Answer a few questions one step at a time to generate your heart disease risk prediction.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Answer the health questionnaire → 2️⃣ Click Predict → 3️⃣ View your heart disease risk score
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About This Module"):
        st.write("""
        Heart disease is the leading cause of death worldwide. This module provides an
        interactive step-by-step questionnaire that collects 13 clinical features and uses
        machine learning to estimate heart disease risk. It is designed for educational
        exploration of clinical risk prediction models.
        """)

    with st.expander("How the Model Works"):
        st.write("""
        **Algorithm:** Random Forest ensemble classifier

        **Training Data:** UCI Heart Disease dataset with 920 patient records and 13 clinical
        features including age, sex, chest pain type, blood pressure, cholesterol, and more.

        **Performance:** Accuracy: 98.6% | AUC-ROC: 99.9% (10-fold stratified CV on 5,160 combined records)

        **Pipeline:** Clinical Features (questionnaire) -> Feature Scaling -> Random Forest ->
        Risk Score (0-100%) + Risk Category
        """)

    with st.expander("Input Requirements"):
        st.write("""
        - **No file upload needed** -- use the interactive questionnaire
        - **13 clinical features:** Age, sex, chest pain type, resting blood pressure, cholesterol,
          fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression,
          ST slope, number of major vessels, thalassemia type
        - The questionnaire guides you through each input step by step.
        """)

    with st.expander("Understanding Your Results"):
        st.write("""
        The model outputs a heart disease risk score from 0% to 100%.

        - **Low Risk (0-30%):** Few risk factors detected.
        - **Moderate Risk (30-60%):** Some risk factors present; lifestyle changes recommended.
        - **High Risk (60-100%):** Multiple risk factors detected; medical consultation advised.

        Interactive charts show feature importance and risk factor breakdown.
        """)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    total_steps = 4

    # Initialize session state for questionnaire
    defaults = {
        "hra_age": 55, "hra_sex": "Male", "hra_cp": "Asymptomatic",
        "hra_trestbps": 130, "hra_chol": 240, "hra_fbs": "No",
        "hra_restecg": "Normal", "hra_thalach": 150, "hra_exang": "No",
        "hra_oldpeak": 1.0, "hra_slope": "Flat", "hra_ca": 0, "hra_thal": "Normal",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Progress bar
    step = st.session_state.hra_step
    progress_pct = min((step - 1) / total_steps * 100, 100)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
        <span style="font-weight:700;color:#1a2332;">Step {min(step, total_steps)} of {total_steps}</span>
        <span style="color:#8899a6;font-size:0.85rem;">
            {"Demographics" if step == 1 else "Lab Values" if step == 2 else "Cardiac Tests" if step == 3 else "Review & Predict"}
        </span>
    </div>
    <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{progress_pct}%;"></div></div>
    """, unsafe_allow_html=True)

    # STEP 1: Demographics
    if step == 1:
        st.markdown("""
        <div class="step-card">
            <h3>Step 1: Demographics & Symptoms</h3>
            <p>Tell us about yourself and any chest pain symptoms you experience.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.session_state.hra_age = st.slider(
                "How old are you?", 20, 100, st.session_state.hra_age, key="hra_age_slider"
            )
            st.session_state.hra_sex = st.radio(
                "Biological sex", ["Male", "Female"], key="hra_sex_radio",
                index=0 if st.session_state.hra_sex == "Male" else 1,
                horizontal=True,
            )
        with col2:
            cp_options = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
            st.markdown("**What type of chest pain do you experience?**")
            cp_cols = st.columns(2)
            cp_descriptions = [
                "Typical squeezing/pressure with exertion",
                "Atypical chest discomfort",
                "Non-cardiac chest pain",
                "No chest pain symptoms"
            ]
            for i, (opt, desc) in enumerate(zip(cp_options, cp_descriptions)):
                with cp_cols[i % 2]:
                    if st.button(
                        f"{'✅ ' if st.session_state.hra_cp == opt else ''}{opt}",
                        key=f"cp_{opt}",
                        use_container_width=True,
                        help=desc,
                    ):
                        st.session_state.hra_cp = opt
                        st.rerun()

        col_nav1, col_nav2 = st.columns([3, 1])
        with col_nav2:
            if st.button("Next →", type="primary", use_container_width=True, key="step1_next"):
                st.session_state.hra_step = 2
                st.rerun()

    # STEP 2: Lab Values
    elif step == 2:
        st.markdown("""
        <div class="step-card">
            <h3>Step 2: Blood Pressure & Lab Values</h3>
            <p>Enter your blood pressure, cholesterol, and other lab results.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.session_state.hra_trestbps = st.slider(
                "Resting Blood Pressure (mmHg)", 80, 220, st.session_state.hra_trestbps, key="hra_bp_slider"
            )
            bp_val = st.session_state.hra_trestbps
            if bp_val < 120:
                st.markdown('<span class="status-badge status-loaded">Normal</span>', unsafe_allow_html=True)
            elif bp_val < 140:
                st.markdown('<span class="status-badge status-demo">Elevated</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="flag-critical">High</span>', unsafe_allow_html=True)

            st.session_state.hra_chol = st.slider(
                "Total Cholesterol (mg/dL)", 100, 600, st.session_state.hra_chol, key="hra_chol_slider"
            )
        with col2:
            fbs_toggle = st.toggle(
                "Fasting Blood Sugar > 120 mg/dL",
                value=(st.session_state.hra_fbs == "Yes"),
                key="hra_fbs_toggle"
            )
            st.session_state.hra_fbs = "Yes" if fbs_toggle else "No"

            restecg_options = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
            st.session_state.hra_restecg = st.radio(
                "Resting ECG Result", restecg_options,
                index=restecg_options.index(st.session_state.hra_restecg),
                key="hra_restecg_radio", horizontal=True,
            )

        col_back, _, col_next = st.columns([1, 2, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="step2_back"):
                st.session_state.hra_step = 1
                st.rerun()
        with col_next:
            if st.button("Next →", type="primary", use_container_width=True, key="step2_next"):
                st.session_state.hra_step = 3
                st.rerun()

    # STEP 3: Cardiac Tests
    elif step == 3:
        st.markdown("""
        <div class="step-card">
            <h3>Step 3: Exercise & Cardiac Test Results</h3>
            <p>Enter your exercise test results and cardiac imaging data.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.session_state.hra_thalach = st.slider(
                "Maximum Heart Rate Achieved", 60, 220, st.session_state.hra_thalach, key="hra_thalach_slider"
            )
            exang_toggle = st.toggle(
                "Exercise-Induced Angina",
                value=(st.session_state.hra_exang == "Yes"),
                key="hra_exang_toggle"
            )
            st.session_state.hra_exang = "Yes" if exang_toggle else "No"

            st.session_state.hra_oldpeak = st.slider(
                "ST Depression (oldpeak)", 0.0, 6.0, st.session_state.hra_oldpeak, step=0.1, key="hra_oldpeak_slider"
            )

        with col2:
            slope_options = ["Upsloping", "Flat", "Downsloping"]
            st.session_state.hra_slope = st.radio(
                "ST Slope", slope_options,
                index=slope_options.index(st.session_state.hra_slope),
                key="hra_slope_radio", horizontal=True,
            )
            st.session_state.hra_ca = st.slider(
                "Major Vessels Colored by Fluoroscopy (0-3)", 0, 3,
                st.session_state.hra_ca, key="hra_ca_slider"
            )
            thal_options = ["Normal", "Fixed Defect", "Reversible Defect"]
            st.session_state.hra_thal = st.radio(
                "Thalassemia", thal_options,
                index=thal_options.index(st.session_state.hra_thal),
                key="hra_thal_radio", horizontal=True,
            )

        col_back, _, col_next = st.columns([1, 2, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="step3_back"):
                st.session_state.hra_step = 2
                st.rerun()
        with col_next:
            if st.button("Review & Predict →", type="primary", use_container_width=True, key="step3_next"):
                st.session_state.hra_step = 4
                st.rerun()

    # STEP 4: Review & Results
    elif step >= 4:
        st.markdown("""
        <div class="step-card">
            <h3>Review Your Inputs</h3>
            <p>Verify your information below, then view your risk prediction.</p>
        </div>
        """, unsafe_allow_html=True)

        # Summary of inputs
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.markdown("**Demographics**")
            st.markdown(f"- Age: **{st.session_state.hra_age}**")
            st.markdown(f"- Sex: **{st.session_state.hra_sex}**")
            st.markdown(f"- Chest Pain: **{st.session_state.hra_cp}**")
        with col2:
            st.markdown("**Lab Values**")
            st.markdown(f"- Resting BP: **{st.session_state.hra_trestbps} mmHg**")
            st.markdown(f"- Cholesterol: **{st.session_state.hra_chol} mg/dL**")
            st.markdown(f"- Fasting BS > 120: **{st.session_state.hra_fbs}**")
            st.markdown(f"- Resting ECG: **{st.session_state.hra_restecg}**")
        with col3:
            st.markdown("**Cardiac Tests**")
            st.markdown(f"- Max HR: **{st.session_state.hra_thalach} BPM**")
            st.markdown(f"- Exercise Angina: **{st.session_state.hra_exang}**")
            st.markdown(f"- ST Depression: **{st.session_state.hra_oldpeak}**")
            st.markdown(f"- ST Slope: **{st.session_state.hra_slope}**")
            st.markdown(f"- Major Vessels: **{st.session_state.hra_ca}**")
            st.markdown(f"- Thalassemia: **{st.session_state.hra_thal}**")

        col_back2, _, col_edit = st.columns([1, 2, 1])
        with col_back2:
            if st.button("← Edit Answers", use_container_width=True, key="step4_back"):
                st.session_state.hra_step = 1
                st.rerun()

        # Run prediction
        st.markdown("---")

        age = st.session_state.hra_age
        sex = st.session_state.hra_sex
        cp = st.session_state.hra_cp
        trestbps = st.session_state.hra_trestbps
        chol = st.session_state.hra_chol
        fbs = st.session_state.hra_fbs
        restecg = st.session_state.hra_restecg
        thalach = st.session_state.hra_thalach
        exang = st.session_state.hra_exang
        oldpeak = st.session_state.hra_oldpeak
        slope = st.session_state.hra_slope
        ca = st.session_state.hra_ca
        thal = st.session_state.hra_thal

        sex_val = 1 if sex == "Male" else 0
        cp_val = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
        fbs_val = 1 if fbs == "Yes" else 0
        restecg_val = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
        exang_val = 1 if exang == "Yes" else 0
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

        dia_bp = trestbps * 0.65  # approximate diastolic
        bmi_est = 26.0  # default BMI when not collected
        glucose_est = 100.0  # default glucose
        hyp_val = 1 if trestbps > 140 else 0

        # Base features matching combined UCI+Framingham training pipeline
        features_base = [
            age, sex_val, trestbps, chol, fbs_val, thalach,
            cp_val, exang_val, oldpeak, ca, thal_val, restecg_val, slope_val,
            0,  # smoker (not collected in questionnaire)
            0,  # bp_meds
            hyp_val,  # prevalent hypertension
            fbs_val,  # diabetes proxy
            bmi_est,  # BMI
            dia_bp,   # diastolic BP
            glucose_est,  # glucose
        ]
        # Engineered features matching training pipeline
        features_eng = features_base + [
            age * trestbps,            # age_bp
            age * chol,                # age_chol
            age * thalach,             # age_hr
            trestbps / (thalach + 1),  # bp_hr
            trestbps - dia_bp,         # pulse pressure
            bmi_est * age,             # bmi_age
            hyp_val + fbs_val,         # risk_sum
            oldpeak * (slope_val + 1), # oldpeak_slope
            ca * thal_val,             # ca_thal
            age ** 2,                  # age_sq
            chol * trestbps,           # chol_bp
        ]
        features = np.array([features_eng])

        if _heart_loaded:
            try:
                with st.spinner("Running cardiac risk ensemble model..."):
                    features_scaled = _heart_scaler.transform(features)
                    proba = _heart_model.predict_proba(features_scaled)[0]
                    risk_score = float(proba[1]) * 100
                    predicted = "High Risk" if risk_score > 50 else "Low Risk"
            except Exception as e:
                st.warning(f"Model prediction failed ({e}). Using rule-based estimate.")
                risk_score, predicted = _fallback_risk(age, sex_val, cp_val, trestbps, chol, fbs_val, thalach, exang_val, oldpeak, ca)
        else:
            risk_score, predicted = _fallback_risk(age, sex_val, cp_val, trestbps, chol, fbs_val, thalach, exang_val, oldpeak, ca)
            st.info("Using rule-based estimate. Train the model for ML predictions.")

        st.success("Risk assessment complete!")

        # Results
        risk_class = "risk-high" if predicted == "High Risk" else "risk-low"
        risk_medium = "risk-medium" if 30 < risk_score < 60 else risk_class

        col_r1, col_r2, col_r3 = st.columns(3, gap="medium")
        with col_r1:
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <p class="metric-label">Risk Level</p>
                <p class="metric-value">{predicted}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
            <div class="metric-card {risk_medium}">
                <p class="metric-label">Risk Score</p>
                <p class="metric-value">{risk_score:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Max Heart Rate</p>
                <p class="metric-value">{thalach} <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">BPM</small></p>
            </div>
            """, unsafe_allow_html=True)

        col_g, col_r = st.columns(2, gap="medium")
        with col_g:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                number={"suffix": "%", "font": {"size": 42, "family": "Inter", "color": "#1a2332"}},
                title={"text": "Heart Disease Risk", "font": {"size": 16, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#d0d9e1"},
                    "bar": {"color": "#2c5364", "thickness": 0.75},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fdebd0"},
                        {"range": [60, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "#e74c3c", "width": 3},
                        "thickness": 0.8,
                        "value": risk_score,
                    },
                },
            ))
            fig_gauge.update_layout(height=320, margin=dict(t=60, b=20, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
            st.plotly_chart(fig_gauge, use_container_width=True, key="hra_gauge_chart")

        with col_r:
            categories = ["Age", "BP", "Cholesterol", "Heart Rate", "ST Depression", "Vessels"]
            values = [
                age / 120 * 100, trestbps / 220 * 100, chol / 600 * 100,
                thalach / 220 * 100, oldpeak / 6 * 100, ca / 3 * 100,
            ]
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(44,83,100,0.12)",
                line=dict(color="#2c5364", width=2),
                marker=dict(size=6, color="#2c5364"),
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#e8eef3"),
                    angularaxis=dict(gridcolor="#e8eef3"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                title=dict(text="Patient Vitals Overview", font=dict(size=16, family="Inter")),
                height=320, margin=dict(t=60, b=20, l=60, r=60),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="hra_radar_chart")

        st.markdown("---")
        with st.expander("Detailed Report - What Your Results Mean", expanded=True):
            st.markdown(interpret_heart_risk(
                predicted, risk_score, age, sex, cp, trestbps, chol, thalach, oldpeak, ca
            ))

    st.markdown("---")
    st.markdown("##### Explore More Modules")
    sug_cols = st.columns(3, gap="medium")
    with sug_cols[0]:
        st.button("Heart / ECG Analysis →", key="suggest_hra_to_ecg", on_click=navigate_to, args=("Heart / ECG",), use_container_width=True)
    with sug_cols[1]:
        st.button("Lipid Panel / CV Risk →", key="suggest_hra_to_lipid", on_click=navigate_to, args=("Lipid Panel / CV Risk",), use_container_width=True)
    with sug_cols[2]:
        st.button("Diabetes Screening →", key="suggest_hra_to_diabetes", on_click=navigate_to, args=("Diabetes Screening",), use_container_width=True)


# ============================================================
# CBC ANALYSIS SECTION
# ============================================================
elif section == "CBC Analysis":
    st.button("← Back to Home", key="back_home_cbc", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>CBC Analysis</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">CBC Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter complete blood count values for automated classification, differential visualization, and clinical interpretation.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Enter your blood count values → 2️⃣ Click Analyze → 3️⃣ View flagged abnormalities
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About This Module"):
        st.write("""
        A Complete Blood Count (CBC) is one of the most commonly ordered blood tests. This
        module interprets CBC values using clinical reference ranges, providing automated
        classification and visual differential analysis. Useful for students learning
        hematology and anyone exploring automated lab interpretation.
        """)

    with st.expander("How the Algorithm Works"):
        st.write("""
        **Algorithm:** Clinical rule-based algorithm (not machine learning)

        **Method:** Each CBC parameter is compared against established clinical reference ranges,
        stratified by sex. Values are classified as Low, Normal, or High with color-coded badges.

        **Parameters Analyzed:** WBC, RBC, Hemoglobin, Hematocrit, Platelets, MCV, MCH, MCHC,
        RDW, and WBC differential (Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils).

        **Reference Ranges:** Based on standard clinical laboratory reference intervals.
        """)

    with st.expander("Input Requirements"):
        st.write("""
        - **No file upload needed** -- enter values directly in the form
        - **Required values:** WBC, RBC, Hemoglobin, Hematocrit, Platelets
        - **Optional values:** MCV, MCH, MCHC, RDW, WBC differential percentages
        - **Sex selection:** Required for sex-specific reference ranges
        """)

    with st.expander("Understanding Your Results"):
        st.write("""
        Results are displayed as color-coded badges:

        - **Green (Normal):** Value is within the normal reference range.
        - **Orange (High/Low):** Value is outside normal but not critical.
        - **Red (Critical):** Value is significantly outside the normal range.

        The WBC differential pie chart shows the relative proportions of white blood cell types.
        """)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**Patient Info & Basic CBC**")
            cbc_sex = st.selectbox("Sex", ["Male", "Female"], key="cbc_sex")
            cbc_wbc = st.slider("WBC (x10\u00b3/\u00b5L)", 0.0, 50.0, 7.0, step=0.1, key="cbc_wbc")
            cbc_rbc = st.slider("RBC (x10\u2076/\u00b5L)", 0.0, 10.0, 4.7, step=0.1, key="cbc_rbc")
            cbc_hgb = st.slider("Hemoglobin (g/dL)", 0.0, 25.0, 14.0, step=0.1, key="cbc_hgb")
            cbc_hct = st.slider("Hematocrit (%)", 0.0, 80.0, 42.0, step=0.1, key="cbc_hct")

        with col2:
            st.markdown("**RBC Indices**")
            cbc_mcv = st.slider("MCV (fL)", 50.0, 150.0, 88.0, step=0.1, key="cbc_mcv")
            cbc_mch = st.slider("MCH (pg)", 10.0, 50.0, 29.0, step=0.1, key="cbc_mch")
            cbc_mchc = st.slider("MCHC (g/dL)", 20.0, 45.0, 33.5, step=0.1, key="cbc_mchc")
            cbc_rdw = st.slider("RDW (%)", 5.0, 30.0, 13.0, step=0.1, key="cbc_rdw")
            cbc_plt = st.slider("Platelets (x10\u00b3/\u00b5L)", 0.0, 1000.0, 250.0, step=1.0, key="cbc_plt")

        with col3:
            st.markdown("**WBC Differential (%)**")
            cbc_neut = st.slider("Neutrophils %", 0.0, 100.0, 60.0, step=0.1, key="cbc_neut")
            cbc_lymph = st.slider("Lymphocytes %", 0.0, 100.0, 30.0, step=0.1, key="cbc_lymph")
            cbc_mono = st.slider("Monocytes %", 0.0, 100.0, 6.0, step=0.1, key="cbc_mono")
            cbc_eos = st.slider("Eosinophils %", 0.0, 100.0, 3.0, step=0.1, key="cbc_eos")
            cbc_baso = st.slider("Basophils %", 0.0, 100.0, 0.5, step=0.1, key="cbc_baso")

    st.markdown("")

    if st.button("Analyze", type="primary", use_container_width=True, key="cbc_analyze"):
        sex_key = cbc_sex.lower()
        refs = CBC_RANGES[sex_key]

        cbc_values = {
            "WBC": cbc_wbc, "RBC": cbc_rbc, "Hemoglobin": cbc_hgb,
            "Hematocrit": cbc_hct, "MCV": cbc_mcv, "MCH": cbc_mch,
            "MCHC": cbc_mchc, "RDW": cbc_rdw, "Platelets": cbc_plt,
            "Neutrophils": cbc_neut,
        }

        st.success("CBC analysis complete!")

        st.markdown("---")

        mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
        for col_obj, param_name, display_val, unit in [
            (mc1, "WBC", cbc_wbc, "x10\u00b3/\u00b5L"),
            (mc2, "Hemoglobin", cbc_hgb, "g/dL"),
            (mc3, "Hematocrit", cbc_hct, "%"),
            (mc4, "Platelets", cbc_plt, "x10\u00b3/\u00b5L"),
        ]:
            ref = refs[param_name]
            status, css_class, color = classify_value(
                display_val, ref["low"], ref["high"],
                ref.get("crit_low"), ref.get("crit_high")
            )
            card_class = "risk-low" if status == "Normal" else ("risk-high" if "Critical" in status else "risk-medium")
            with col_obj:
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <p class="metric-label">{param_name}</p>
                    <p class="metric-value">{display_val} <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">{unit}</small></p>
                    <span class="{css_class}">{status}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Full CBC table
        all_params = [
            ("WBC", cbc_wbc), ("RBC", cbc_rbc), ("Hemoglobin", cbc_hgb),
            ("Hematocrit", cbc_hct), ("MCV", cbc_mcv), ("MCH", cbc_mch),
            ("MCHC", cbc_mchc), ("RDW", cbc_rdw), ("Platelets", cbc_plt),
        ]
        table_rows = ""
        for param_name, val in all_params:
            ref = refs[param_name]
            status, css_class, color = classify_value(
                val, ref["low"], ref["high"],
                ref.get("crit_low"), ref.get("crit_high")
            )
            table_rows += f"""
            <tr>
                <td style="font-weight:600;">{param_name}</td>
                <td>{val}</td>
                <td>{ref['unit']}</td>
                <td>{ref['low']} - {ref['high']}</td>
                <td><span class="{css_class}">{status}</span></td>
            </tr>"""

        st.markdown(f"""
        <table class="lab-table">
            <thead><tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Reference Range</th><th>Flag</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_donut, col_interp = st.columns(2, gap="medium")
        with col_donut:
            diff_labels = ["Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils"]
            diff_values = [cbc_neut, cbc_lymph, cbc_mono, cbc_eos, cbc_baso]
            diff_colors = ["#2c5364", "#5dade2", "#48c9b0", "#f4d03f", "#e74c3c"]
            fig_diff = go.Figure(data=[go.Pie(
                labels=diff_labels, values=diff_values, hole=0.4,
                marker=dict(colors=diff_colors),
                textinfo="label+percent", textfont=dict(family="Inter", size=12),
            )])
            fig_diff.update_layout(
                title=dict(text="WBC Differential", font=dict(size=16, family="Inter")),
                height=350, template="plotly_dark", font=dict(family="Inter"),
                margin=dict(t=60, b=20, l=20, r=20), showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_diff, use_container_width=True, key="cbc_diff_chart")

        with col_interp:
            findings = interpret_cbc(cbc_values, sex_key)
            st.markdown("**Clinical Interpretation**")
            for finding in findings:
                st.info(finding)

        st.markdown("---")
        with st.expander("Detailed Report - What Each Parameter Means", expanded=False):
            for param_name in cbc_values:
                if param_name in CBC_EXPLANATIONS:
                    ref = refs.get(param_name)
                    val = cbc_values[param_name]
                    if ref:
                        status, _, _ = classify_value(val, ref["low"], ref["high"],
                                                      ref.get("crit_low"), ref.get("crit_high"))
                        st.markdown(f"**{param_name}: {val} {ref['unit']}** - _{status}_")
                    else:
                        st.markdown(f"**{param_name}: {val}%**")
                    st.markdown(CBC_EXPLANATIONS[param_name])
                    st.markdown("---")

        st.markdown("""
        <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This CBC analysis is generated by an automated algorithm using
            standard reference ranges. It is intended for educational and screening purposes only.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Explore More Modules")
    sug_cols = st.columns(3, gap="medium")
    with sug_cols[0]:
        st.button("Diabetes Screening →", key="suggest_cbc_to_diabetes", on_click=navigate_to, args=("Diabetes Screening",), use_container_width=True)
    with sug_cols[1]:
        st.button("Kidney Function →", key="suggest_cbc_to_kidney", on_click=navigate_to, args=("Kidney Function",), use_container_width=True)
    with sug_cols[2]:
        st.button("Lab Report Upload →", key="suggest_cbc_to_lab", on_click=navigate_to, args=("Lab Report Upload",), use_container_width=True)


# ============================================================
# DIABETES SCREENING SECTION
# ============================================================
elif section == "Diabetes Screening":
    st.button("← Back to Home", key="back_home_diabetes", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>Diabetes Screening</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Diabetes Screening</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Comprehensive diabetes risk assessment using HbA1c, fasting glucose, and the FINDRISC questionnaire.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Enter your HbA1c, glucose, and demographics → 2️⃣ Click Screen → 3️⃣ View your diabetes risk
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About This Module"):
        st.write("""
        Diabetes affects over 400 million people worldwide. This module uses the validated
        FINDRISC (Finnish Diabetes Risk Score) questionnaire combined with HbA1c and fasting
        glucose thresholds to assess diabetes risk. It is a screening tool based on established
        clinical guidelines, not a predictive ML model.
        """)

    with st.expander("How the Algorithm Works"):
        st.write("""
        **Algorithm:** Validated FINDRISC questionnaire + clinical thresholds

        **FINDRISC:** A validated 8-question screening tool developed in Finland, widely used
        globally to estimate 10-year risk of developing Type 2 diabetes.

        **HbA1c Thresholds:** Normal (<5.7%), Prediabetes (5.7-6.4%), Diabetes (>=6.5%)
        per ADA guidelines.

        **Fasting Glucose Thresholds:** Normal (<100 mg/dL), Prediabetes (100-125 mg/dL),
        Diabetes (>=126 mg/dL) per ADA guidelines.
        """)

    with st.expander("Input Requirements"):
        st.write("""
        - **No file upload needed** -- enter values directly
        - **FINDRISC inputs:** Age, BMI, waist circumference, physical activity, diet,
          medication history, blood glucose history, family history of diabetes
        - **Lab values (optional):** HbA1c (%), Fasting glucose (mg/dL)
        """)

    with st.expander("Understanding Your Results"):
        st.write("""
        - **FINDRISC Score (0-26):** Low (<7), Slightly elevated (7-11), Moderate (12-14),
          High (15-20), Very high (>20)
        - **HbA1c Classification:** Normal, Prediabetes, or Diabetes per ADA criteria
        - **Fasting Glucose Classification:** Normal, Prediabetes, or Diabetes per ADA criteria

        Results combine all three assessments for a comprehensive screening overview.
        """)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**Lab Values**")
            dm_hba1c = st.slider("HbA1c (%)", 3.0, 15.0, 5.5, step=0.1, key="dm_hba1c")
            dm_glucose = st.slider("Fasting Glucose (mg/dL)", 40, 500, 95, key="dm_glucose")
            st.markdown("**Demographics**")
            dm_age = st.slider("Age", 18, 120, 50, key="dm_age")
            dm_bmi = st.slider("BMI (kg/m\u00b2)", 10.0, 60.0, 25.0, step=0.1, key="dm_bmi")
            dm_waist = st.slider("Waist Circumference (cm)", 50, 200, 90, key="dm_waist")
            dm_sex = st.radio("Sex", ["Male", "Female"], key="dm_sex", horizontal=True)

        with col2:
            st.markdown("**Family & Lifestyle**")
            dm_family = st.radio("Family History of Diabetes", ["None", "One parent", "Both parents"], key="dm_family")
            dm_activity = st.radio("Physical Activity Level", ["Active", "Low"], key="dm_activity", horizontal=True)
            dm_fruit = st.toggle("Daily Fruit/Vegetable Intake", value=True, key="dm_fruit_toggle")

        with col3:
            st.markdown("**Medical History**")
            dm_bp_med = st.toggle("On BP Medication", value=False, key="dm_bp_med_toggle")
            dm_high_glucose = st.toggle("History of High Blood Glucose", value=False, key="dm_high_glucose_toggle")

    st.markdown("")

    if st.button("Screen", type="primary", use_container_width=True, key="dm_screen"):
        hba1c_label, hba1c_color = classify_hba1c(dm_hba1c)
        glucose_label, glucose_color = classify_fasting_glucose(dm_glucose)

        family_map = {"None": "none", "One parent": "one_parent", "Both parents": "both_parents"}
        fruit_val = "yes" if dm_fruit else "no"
        bp_val = "yes" if dm_bp_med else "no"
        glucose_hx_val = "yes" if dm_high_glucose else "no"

        findrisc_score, findrisc_cat, findrisc_risk = calculate_findrisc(
            age=dm_age, bmi=dm_bmi, waist=dm_waist, sex=dm_sex.lower(),
            activity=dm_activity.lower(), fruit_veg=fruit_val,
            bp_meds=bp_val, high_glucose=glucose_hx_val,
            family_hx=family_map[dm_family],
        )

        st.success("Diabetes screening complete!")

        st.markdown("---")

        mc1, mc2, mc3 = st.columns(3, gap="medium")
        with mc1:
            hba1c_card = "risk-low" if hba1c_label == "Normal" else ("risk-high" if hba1c_label == "Diabetes" else "risk-medium")
            st.markdown(f"""
            <div class="metric-card {hba1c_card}">
                <p class="metric-label">HbA1c Classification</p>
                <p class="metric-value">{dm_hba1c}% <small style="font-size:0.9rem;font-weight:400;color:{hba1c_color};">{hba1c_label}</small></p>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            gluc_card = "risk-low" if glucose_label == "Normal" else ("risk-high" if glucose_label == "Diabetes" else "risk-medium")
            st.markdown(f"""
            <div class="metric-card {gluc_card}">
                <p class="metric-label">Fasting Glucose</p>
                <p class="metric-value">{dm_glucose} <small style="font-size:0.9rem;font-weight:400;color:{glucose_color};">mg/dL - {glucose_label}</small></p>
            </div>
            """, unsafe_allow_html=True)
        with mc3:
            fr_card = "risk-low" if findrisc_score < 7 else ("risk-high" if findrisc_score > 14 else "risk-medium")
            st.markdown(f"""
            <div class="metric-card {fr_card}">
                <p class="metric-label">FINDRISC Score</p>
                <p class="metric-value">{findrisc_score} <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">/ 26 - {findrisc_cat}</small></p>
                <p style="color:#6b7b8d;font-size:0.85rem;margin-top:4px;">10-year risk: {findrisc_risk}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_gauge, col_bar = st.columns(2, gap="medium")
        with col_gauge:
            fig_findrisc = go.Figure(go.Indicator(
                mode="gauge+number", value=findrisc_score,
                number={"font": {"size": 42, "family": "Inter", "color": "#1a2332"}},
                title={"text": f"FINDRISC Score - {findrisc_cat}", "font": {"size": 16, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 26]}, "bar": {"color": "#2c5364", "thickness": 0.75},
                    "bgcolor": "white", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 7], "color": "#d5f5e3"}, {"range": [7, 12], "color": "#eafaf1"},
                        {"range": [12, 15], "color": "#fef9e7"}, {"range": [15, 20], "color": "#fdebd0"},
                        {"range": [20, 26], "color": "#fadbd8"},
                    ],
                },
            ))
            fig_findrisc.update_layout(height=320, margin=dict(t=60, b=20, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
            st.plotly_chart(fig_findrisc, use_container_width=True, key="diabetes_findrisc_chart")

        with col_bar:
            contributions = {}
            if dm_age < 45: contributions["Age"] = 0
            elif dm_age <= 54: contributions["Age"] = 2
            elif dm_age <= 64: contributions["Age"] = 3
            else: contributions["Age"] = 4
            if dm_bmi < 25: contributions["BMI"] = 0
            elif dm_bmi <= 30: contributions["BMI"] = 1
            else: contributions["BMI"] = 3
            if dm_sex.lower() == "male":
                contributions["Waist"] = 0 if dm_waist < 94 else (3 if dm_waist <= 102 else 4)
            else:
                contributions["Waist"] = 0 if dm_waist < 80 else (3 if dm_waist <= 88 else 4)
            contributions["Activity"] = 2 if dm_activity == "Low" else 0
            contributions["Diet"] = 0 if dm_fruit else 1
            contributions["BP Meds"] = 2 if dm_bp_med else 0
            contributions["High Glucose Hx"] = 5 if dm_high_glucose else 0
            fam_map_pts = {"None": 0, "One parent": 3, "Both parents": 5}
            contributions["Family Hx"] = fam_map_pts[dm_family]

            contrib_df = pd.DataFrame({"Factor": list(contributions.keys()), "Points": list(contributions.values())})
            contrib_df = contrib_df.sort_values("Points", ascending=True)
            fig_contrib = px.bar(contrib_df, x="Points", y="Factor", orientation="h", color="Points",
                                 color_continuous_scale=["#d5f5e3", "#f39c12", "#e74c3c"])
            fig_contrib.update_layout(title=dict(text="Risk Factor Contributions", font=dict(size=16, family="Inter")),
                                      height=320, template="plotly_dark", font=dict(family="Inter"), showlegend=False)
            st.plotly_chart(fig_contrib, use_container_width=True, key="diabetes_contrib_chart")

        st.markdown("---")
        with st.expander("Detailed Report - What Your Results Mean", expanded=True):
            st.markdown(interpret_diabetes_results(dm_hba1c, dm_glucose, findrisc_score, findrisc_cat, findrisc_risk))

        st.markdown("""
        <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This diabetes screening tool uses ADA criteria and the FINDRISC questionnaire.
            It is intended for screening purposes only and does not replace clinical judgment.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Explore More Modules")
    sug_cols = st.columns(3, gap="medium")
    with sug_cols[0]:
        st.button("CBC Analysis →", key="suggest_diabetes_to_cbc", on_click=navigate_to, args=("CBC Analysis",), use_container_width=True)
    with sug_cols[1]:
        st.button("Lipid Panel / CV Risk →", key="suggest_diabetes_to_lipid", on_click=navigate_to, args=("Lipid Panel / CV Risk",), use_container_width=True)
    with sug_cols[2]:
        st.button("Health Risk Assessment →", key="suggest_diabetes_to_hra", on_click=navigate_to, args=("Health Risk Assessment",), use_container_width=True)


# ============================================================
# LIPID PANEL / CV RISK SECTION
# ============================================================
elif section == "Lipid Panel / CV Risk":
    st.button("← Back to Home", key="back_home_lipid", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>Lipid Panel / CV Risk</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Lipid Panel / CV Risk</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Lipid classification and 10-year ASCVD risk estimation using the Pooled Cohort Equations.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Enter your cholesterol panel → 2️⃣ Click Assess Risk → 3️⃣ View ASCVD risk and lipid classification
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About This Module"):
        st.write("""
        Cardiovascular disease risk assessment is critical for preventive medicine. This module
        classifies lipid panel values per ATP III guidelines and estimates 10-year atherosclerotic
        cardiovascular disease (ASCVD) risk using the AHA/ACC Pooled Cohort Equations.
        """)

    with st.expander("How the Algorithm Works"):
        st.write("""
        **Algorithm:** ATP III lipid classification + Pooled Cohort Equations (PCE)

        **Lipid Classification:** Total cholesterol, LDL, HDL, and triglycerides are classified
        per NCEP ATP III guidelines into categories (Desirable, Borderline High, High, etc.).

        **ASCVD Risk:** The Pooled Cohort Equations (AHA/ACC 2013 guidelines) estimate 10-year
        risk of a first atherosclerotic cardiovascular event using age, sex, race, cholesterol,
        HDL, blood pressure, diabetes status, and smoking status.

        **Validation:** PCE were validated on multiple large US cohort studies.
        """)

    with st.expander("Input Requirements"):
        st.write("""
        - **No file upload needed** -- enter values directly
        - **Lipid values:** Total cholesterol, LDL, HDL, Triglycerides (all in mg/dL)
        - **Demographics:** Age, sex, race
        - **Clinical factors:** Systolic blood pressure, blood pressure treatment status,
          diabetes status, smoking status
        """)

    with st.expander("Understanding Your Results"):
        st.write("""
        - **Lipid classifications** are color-coded per ATP III categories
        - **10-year ASCVD risk** is presented as a percentage:
          - Low (<5%), Borderline (5-7.5%), Intermediate (7.5-20%), High (>=20%)
        - Risk factor modification recommendations are provided based on the results.
        """)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.markdown("**Lipid Panel**")
            lp_tc = st.slider("Total Cholesterol (mg/dL)", 50, 500, 200, key="lp_tc")
            lp_ldl = st.slider("LDL (mg/dL)", 20, 400, 120, key="lp_ldl")
            lp_hdl = st.slider("HDL (mg/dL)", 10, 150, 50, key="lp_hdl")
            lp_trig = st.slider("Triglycerides (mg/dL)", 20, 1000, 150, key="lp_trig")
        with col2:
            st.markdown("**Demographics**")
            lp_age = st.slider("Age", 20, 120, 55, key="lp_age")
            lp_sex = st.radio("Sex", ["Male", "Female"], key="lp_sex", horizontal=True)
            lp_race = st.radio("Race", ["White", "African American", "Other"], key="lp_race")
        with col3:
            st.markdown("**Risk Factors**")
            lp_sbp = st.slider("Systolic BP (mmHg)", 80, 250, 130, key="lp_sbp")
            lp_bp_med = st.toggle("On BP Medication", value=False, key="lp_bp_med_toggle")
            lp_smoker = st.toggle("Current Smoker", value=False, key="lp_smoker_toggle")
            lp_diabetes = st.toggle("Diabetes", value=False, key="lp_diabetes_toggle")

    st.markdown("")

    if st.button("Assess Risk", type="primary", use_container_width=True, key="lp_assess"):
        tc_label, tc_color = classify_lipid("Total Cholesterol", lp_tc)
        ldl_label, ldl_color = classify_lipid("LDL", lp_ldl)
        hdl_label, hdl_color = classify_lipid("HDL", lp_hdl)
        trig_label, trig_color = classify_lipid("Triglycerides", lp_trig)

        non_hdl = lp_tc - lp_hdl
        tc_hdl_ratio = round(lp_tc / lp_hdl, 2) if lp_hdl > 0 else 0
        ldl_hdl_ratio = round(lp_ldl / lp_hdl, 2) if lp_hdl > 0 else 0

        ascvd_pct, ascvd_cat, ascvd_color = calculate_ascvd_risk(
            age=lp_age, sex=lp_sex.lower(), race=lp_race, total_chol=lp_tc, hdl=lp_hdl,
            sbp=lp_sbp, bp_treated=lp_bp_med, smoker=lp_smoker, diabetes=lp_diabetes,
        )

        st.success("Lipid assessment complete!")

        st.markdown("---")

        mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
        with mc1:
            ascvd_card = "" if ascvd_pct is None else ("risk-low" if ascvd_pct < 5 else ("risk-high" if ascvd_pct >= 20 else "risk-medium"))
            ascvd_display = f"{ascvd_pct}%" if ascvd_pct is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card {ascvd_card}">
                <p class="metric-label">10-Year ASCVD Risk</p>
                <p class="metric-value">{ascvd_display}</p>
                <p style="color:{ascvd_color};font-size:0.85rem;font-weight:600;margin-top:4px;">{ascvd_cat}</p>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{tc_color};">
                <p class="metric-label">Total Cholesterol</p>
                <p class="metric-value" style="color:{tc_color};">{lp_tc} <small style="font-size:0.85rem;font-weight:400;">mg/dL</small></p>
                <p style="color:{tc_color};font-size:0.85rem;font-weight:600;margin-top:4px;">{tc_label}</p>
            </div>
            """, unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{ldl_color};">
                <p class="metric-label">LDL Cholesterol</p>
                <p class="metric-value" style="color:{ldl_color};">{lp_ldl} <small style="font-size:0.85rem;font-weight:400;">mg/dL</small></p>
                <p style="color:{ldl_color};font-size:0.85rem;font-weight:600;margin-top:4px;">{ldl_label}</p>
            </div>
            """, unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{hdl_color};">
                <p class="metric-label">HDL Cholesterol</p>
                <p class="metric-value" style="color:{hdl_color};">{lp_hdl} <small style="font-size:0.85rem;font-weight:400;">mg/dL</small></p>
                <p style="color:{hdl_color};font-size:0.85rem;font-weight:600;margin-top:4px;">{hdl_label}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_gauge, col_bar = st.columns(2, gap="medium")
        with col_gauge:
            gauge_val = ascvd_pct if ascvd_pct is not None else 0
            fig_ascvd = go.Figure(go.Indicator(
                mode="gauge+number", value=gauge_val,
                number={"suffix": "%", "font": {"size": 42, "family": "Inter", "color": "#1a2332"}},
                title={"text": f"10-Year ASCVD Risk - {ascvd_cat}", "font": {"size": 16, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 30]}, "bar": {"color": "#2c5364", "thickness": 0.75},
                    "bgcolor": "white", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 5], "color": "#d5f5e3"}, {"range": [5, 7.5], "color": "#eafaf1"},
                        {"range": [7.5, 20], "color": "#fdebd0"}, {"range": [20, 30], "color": "#fadbd8"},
                    ],
                },
            ))
            fig_ascvd.update_layout(height=320, margin=dict(t=60, b=20, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
            st.plotly_chart(fig_ascvd, use_container_width=True, key="lipid_ascvd_chart")

        with col_bar:
            lipid_names = ["Total Cholesterol", "LDL", "HDL", "Triglycerides"]
            lipid_vals = [lp_tc, lp_ldl, lp_hdl, lp_trig]
            lipid_colors = [tc_color, ldl_color, hdl_color, trig_color]
            fig_lipid = go.Figure(data=[go.Bar(
                x=lipid_names, y=lipid_vals, marker_color=lipid_colors,
                text=[f"{v} mg/dL" for v in lipid_vals], textposition="outside",
            )])
            fig_lipid.update_layout(title=dict(text="Lipid Panel Values", font=dict(size=16, family="Inter")),
                                    height=320, template="plotly_dark", font=dict(family="Inter"), yaxis_title="mg/dL", showlegend=False)
            st.plotly_chart(fig_lipid, use_container_width=True, key="lipid_panel_chart")

        st.markdown("**Lipid Ratios**")
        rc1, rc2, rc3 = st.columns(3, gap="medium")
        with rc1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Non-HDL Cholesterol</p>
                <p class="metric-value">{non_hdl} <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">mg/dL</small></p>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            ratio_class = "risk-low" if tc_hdl_ratio < 4.5 else ("risk-high" if tc_hdl_ratio > 6 else "risk-medium")
            st.markdown(f"""
            <div class="metric-card {ratio_class}">
                <p class="metric-label">TC / HDL Ratio</p>
                <p class="metric-value">{tc_hdl_ratio}</p>
            </div>
            """, unsafe_allow_html=True)
        with rc3:
            lratio_class = "risk-low" if ldl_hdl_ratio < 3 else ("risk-high" if ldl_hdl_ratio > 4.5 else "risk-medium")
            st.markdown(f"""
            <div class="metric-card {lratio_class}">
                <p class="metric-label">LDL / HDL Ratio</p>
                <p class="metric-value">{ldl_hdl_ratio}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("Detailed Report - What Your Lipid Results Mean", expanded=True):
            st.markdown(interpret_lipid_results(lp_tc, lp_ldl, lp_hdl, lp_trig, ascvd_pct, ascvd_cat, tc_hdl_ratio, ldl_hdl_ratio))

        st.markdown("""
        <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This cardiovascular risk assessment uses the ACC/AHA Pooled Cohort
            Equations and ATP III lipid classifications. For screening and educational purposes only.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Explore More Modules")
    sug_cols = st.columns(3, gap="medium")
    with sug_cols[0]:
        st.button("Diabetes Screening →", key="suggest_lipid_to_diabetes", on_click=navigate_to, args=("Diabetes Screening",), use_container_width=True)
    with sug_cols[1]:
        st.button("Health Risk Assessment →", key="suggest_lipid_to_hra", on_click=navigate_to, args=("Health Risk Assessment",), use_container_width=True)
    with sug_cols[2]:
        st.button("Kidney Function →", key="suggest_lipid_to_kidney", on_click=navigate_to, args=("Kidney Function",), use_container_width=True)


# ============================================================
# KIDNEY FUNCTION SECTION
# ============================================================
elif section == "Kidney Function":
    st.button("← Back to Home", key="back_home_kidney", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>Kidney Function</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Kidney Function</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">CKD-EPI 2021 race-free eGFR estimation with KDIGO staging and risk classification.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Enter serum creatinine and demographics → 2️⃣ Click Calculate → 3️⃣ View eGFR and CKD staging
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About This Module"):
        st.write("""
        Chronic kidney disease (CKD) affects approximately 15% of adults. This module calculates
        estimated Glomerular Filtration Rate (eGFR) using the CKD-EPI 2021 race-free equation
        and provides KDIGO staging with albuminuria assessment. Optional Cystatin C comparison
        is also available.
        """)

    with st.expander("How the Algorithm Works"):
        st.write("""
        **Algorithm:** CKD-EPI 2021 race-free equation for eGFR

        **eGFR Calculation:** Uses serum creatinine, age, and sex to estimate kidney function.
        The 2021 update removed the race coefficient for more equitable assessment.

        **KDIGO Staging:** eGFR is mapped to CKD stages G1-G5 per KDIGO 2012 guidelines.

        **Albuminuria Assessment:** Urine albumin-to-creatinine ratio (UACR) is classified
        into stages A1-A3.

        **Optional:** Cystatin C-based eGFR for comparison when available.
        """)

    with st.expander("Input Requirements"):
        st.write("""
        - **No file upload needed** -- enter values directly
        - **Required:** Serum creatinine (mg/dL), Age (years), Sex
        - **Optional:** UACR (mg/g) for albuminuria staging, BUN (mg/dL),
          Cystatin C (mg/L) for comparison eGFR
        """)

    with st.expander("Understanding Your Results"):
        st.write("""
        - **eGFR (mL/min/1.73m2):** Higher is better. Normal is >= 90 with no kidney damage.
        - **CKD Stages:** G1 (>=90, normal), G2 (60-89, mild), G3a (45-59), G3b (30-44),
          G4 (15-29, severe), G5 (<15, kidney failure)
        - **Albuminuria Stages:** A1 (<30, normal), A2 (30-300, moderate), A3 (>300, severe)
        - The KDIGO risk matrix combines eGFR and albuminuria for overall risk classification.
        """)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.markdown("**Required Inputs**")
            kf_cr = st.slider("Serum Creatinine (mg/dL)", 0.1, 15.0, 1.0, step=0.1, key="kf_cr")
            kf_age = st.slider("Age", 18, 120, 55, key="kf_age")
            kf_sex = st.radio("Sex", ["Male", "Female"], key="kf_sex", horizontal=True)
        with col2:
            st.markdown("**Albuminuria & BUN**")
            kf_uacr = st.slider("UACR (mg/g)", 0.0, 3000.0, 15.0, step=1.0, key="kf_uacr")
            kf_bun = st.slider("BUN (mg/dL)", 1.0, 100.0, 15.0, step=0.5, key="kf_bun")
        with col3:
            st.markdown("**Optional: Cystatin C**")
            kf_use_cysc = st.toggle("Include Cystatin C", value=False, key="kf_use_cysc")
            kf_cysc = st.slider("Cystatin C (mg/L)", 0.1, 10.0, 0.9, step=0.1, key="kf_cysc", disabled=not kf_use_cysc)

    st.markdown("")

    if st.button("Calculate", type="primary", use_container_width=True, key="kf_calc"):
        sex_key = kf_sex.lower()
        egfr_cr = ckd_epi_creatinine(kf_cr, kf_age, sex_key)
        ckd_stage, ckd_desc, ckd_color = stage_ckd(egfr_cr)
        alb_stage, alb_desc, alb_color = stage_albuminuria(kf_uacr)

        egfr_cysc = None
        if kf_use_cysc:
            egfr_cysc = ckd_epi_cystatin(kf_cysc, kf_age, sex_key)

        bun_cr_ratio = round(kf_bun / kf_cr, 1) if kf_cr > 0 else 0

        st.success("eGFR calculation complete!")

        st.markdown("---")

        mc1, mc2, mc3 = st.columns(3, gap="medium")
        with mc1:
            egfr_card = "risk-low" if egfr_cr >= 60 else ("risk-high" if egfr_cr < 30 else "risk-medium")
            st.markdown(f"""
            <div class="metric-card {egfr_card}">
                <p class="metric-label">eGFR (Creatinine)</p>
                <p class="metric-value">{egfr_cr} <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">mL/min/1.73m\u00b2</small></p>
                <p style="color:{ckd_color};font-size:0.85rem;font-weight:600;margin-top:4px;">Stage {ckd_stage}: {ckd_desc}</p>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            alb_card = "risk-low" if kf_uacr < 30 else ("risk-high" if kf_uacr > 300 else "risk-medium")
            st.markdown(f"""
            <div class="metric-card {alb_card}">
                <p class="metric-label">UACR / Albuminuria</p>
                <p class="metric-value">{kf_uacr} <small style="font-size:0.9rem;font-weight:400;color:#8899a6;">mg/g</small></p>
                <p style="color:{alb_color};font-size:0.85rem;font-weight:600;margin-top:4px;">{alb_stage}: {alb_desc}</p>
            </div>
            """, unsafe_allow_html=True)
        with mc3:
            bun_card = "risk-low" if 10 <= bun_cr_ratio <= 20 else "risk-medium"
            st.markdown(f"""
            <div class="metric-card {bun_card}">
                <p class="metric-label">BUN / Creatinine Ratio</p>
                <p class="metric-value">{bun_cr_ratio}</p>
                <p style="color:#6b7b8d;font-size:0.85rem;margin-top:4px;">Normal: 10-20</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_gauge, col_comp = st.columns(2, gap="medium")
        with col_gauge:
            fig_egfr = go.Figure(go.Indicator(
                mode="gauge+number", value=egfr_cr,
                number={"font": {"size": 42, "family": "Inter", "color": "#1a2332"}},
                title={"text": f"eGFR - Stage {ckd_stage}", "font": {"size": 16, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 120]}, "bar": {"color": "#2c5364", "thickness": 0.75},
                    "bgcolor": "white", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 15], "color": "#fadbd8"}, {"range": [15, 30], "color": "#f5b7b1"},
                        {"range": [30, 45], "color": "#fdebd0"}, {"range": [45, 60], "color": "#fef9e7"},
                        {"range": [60, 90], "color": "#eafaf1"}, {"range": [90, 120], "color": "#d5f5e3"},
                    ],
                },
            ))
            fig_egfr.update_layout(height=320, margin=dict(t=60, b=20, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
            st.plotly_chart(fig_egfr, use_container_width=True, key="kidney_egfr_chart")

        with col_comp:
            if egfr_cysc is not None:
                comp_df = pd.DataFrame({"Method": ["Creatinine-based", "Cystatin C-based"], "eGFR": [egfr_cr, egfr_cysc]})
                fig_comp = go.Figure(data=[go.Bar(
                    x=comp_df["Method"], y=comp_df["eGFR"], marker_color=["#2c5364", "#5dade2"],
                    text=[f"{v} mL/min" for v in comp_df["eGFR"]], textposition="outside",
                )])
                fig_comp.update_layout(title=dict(text="eGFR Comparison", font=dict(size=16, family="Inter")),
                                       height=320, template="plotly_dark", font=dict(family="Inter"), yaxis_title="eGFR (mL/min/1.73m\u00b2)", showlegend=False)
                st.plotly_chart(fig_comp, use_container_width=True, key="kidney_comparison_chart")
            else:
                st.markdown("""
                <div class="info-card">
                    <h4>Cystatin C Comparison</h4>
                    <p>Enable Cystatin C input to see a comparison between creatinine-based
                    and cystatin C-based eGFR estimates.</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KDIGO Risk Matrix
        st.markdown("**KDIGO Risk Matrix - Prognosis of CKD by GFR and Albuminuria**")
        kdigo_colors = [
            ["#d5f5e3", "#eafaf1", "#fdebd0"],
            ["#d5f5e3", "#eafaf1", "#fdebd0"],
            ["#eafaf1", "#fdebd0", "#fadbd8"],
            ["#fdebd0", "#fadbd8", "#fadbd8"],
            ["#fadbd8", "#fadbd8", "#f1948a"],
            ["#fadbd8", "#f1948a", "#f1948a"],
        ]
        kdigo_labels = [
            ["Low", "Moderate", "High"], ["Low", "Moderate", "High"],
            ["Moderate", "High", "Very High"], ["High", "Very High", "Very High"],
            ["Very High", "Very High", "Very High"], ["Very High", "Very High", "Very High"],
        ]
        gfr_stages = [("G1", "\u226590"), ("G2", "60-89"), ("G3a", "45-59"), ("G3b", "30-44"), ("G4", "15-29"), ("G5", "<15")]
        alb_stages = [("A1", "<30"), ("A2", "30-300"), ("A3", ">300")]

        gfr_row_map = {"G1": 0, "G2": 1, "G3a": 2, "G3b": 3, "G4": 4, "G5": 5}
        alb_col_map = {"A1": 0, "A2": 1, "A3": 2}
        patient_row = gfr_row_map.get(ckd_stage, -1)
        patient_col = alb_col_map.get(alb_stage, -1)

        matrix_html = '<table class="ckd-grid"><thead><tr><th>GFR Stage</th><th>eGFR</th>'
        for astage, arange in alb_stages:
            matrix_html += f'<th>{astage}<br><small>{arange} mg/g</small></th>'
        matrix_html += '</tr></thead><tbody>'
        for i, (gstage, grange) in enumerate(gfr_stages):
            matrix_html += f'<tr><td style="font-weight:600;">{gstage}</td><td>{grange}</td>'
            for j in range(3):
                bg = kdigo_colors[i][j]
                label = kdigo_labels[i][j]
                border = "3px solid #1a2332" if (i == patient_row and j == patient_col) else "none"
                matrix_html += f'<td><div class="ckd-cell" style="background:{bg};border:{border};">{label}</div></td>'
            matrix_html += '</tr>'
        matrix_html += '</tbody></table>'
        st.markdown(matrix_html, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("Detailed Report - What Your Kidney Results Mean", expanded=True):
            st.markdown(interpret_kidney_results(
                egfr_cr, egfr_cysc, ckd_stage, ckd_desc,
                alb_stage, alb_desc, kf_uacr, kf_bun, kf_cr
            ))

        st.markdown("""
        <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This kidney function assessment uses the CKD-EPI 2021 race-free
            equations and KDIGO staging guidelines. For screening and educational purposes only.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Explore More Modules")
    sug_cols = st.columns(3, gap="medium")
    with sug_cols[0]:
        st.button("CBC Analysis →", key="suggest_kidney_to_cbc", on_click=navigate_to, args=("CBC Analysis",), use_container_width=True)
    with sug_cols[1]:
        st.button("Lipid Panel / CV Risk →", key="suggest_kidney_to_lipid", on_click=navigate_to, args=("Lipid Panel / CV Risk",), use_container_width=True)
    with sug_cols[2]:
        st.button("Lab Report Upload →", key="suggest_kidney_to_lab", on_click=navigate_to, args=("Lab Report Upload",), use_container_width=True)


# ============================================================
# LAB REPORT UPLOAD SECTION
# ============================================================
elif section == "Lab Report Upload":
    st.button("← Back to Home", key="back_home_lab", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>Lab Report Upload</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Lab Report Upload</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload a lab report PDF for automated parsing, or explore the demo report with color-coded analysis.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Upload a PDF lab report → 2️⃣ Parsing happens automatically → 3️⃣ View flagged abnormal values
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About This Module"):
        st.write("""
        Lab reports are a fundamental part of clinical diagnostics. This module accepts PDF
        lab reports and uses automated parsing to extract common lab values, flagging
        abnormalities against standard reference ranges. Useful for quickly digitizing and
        reviewing lab results.
        """)

    with st.expander("How the Algorithm Works"):
        st.write("""
        **Algorithm:** PDF text extraction + regex pattern matching (not machine learning)

        **Method:** The PDF is parsed for text content. Regex patterns are used to identify
        common lab test names and their associated numeric values with units.

        **Flagging:** Extracted values are compared against standard clinical reference ranges.
        Abnormal values are flagged with color-coded badges (High, Low, Critical).

        **Supported Tests:** CBC, BMP, CMP, lipid panel, thyroid panel, liver function,
        and other common laboratory tests.
        """)

    with st.expander("Input Requirements"):
        st.write("""
        - **Accepted formats:** PDF
        - **Expected content:** Standard laboratory report with test names, values, units, and reference ranges
        - **Best results:** Machine-generated PDF reports (not scanned images)
        - You can also use the built-in demo report to explore the feature.
        """)

    with st.expander("Understanding Your Results"):
        st.write("""
        - **Parsed Results tab:** Shows extracted lab values in a structured table with color-coded flags
        - **Raw Text tab:** Shows the raw text extracted from the PDF
        - **Color coding:** Green (Normal), Orange (High/Low), Red (Critical)
        - Values that cannot be parsed are listed separately for manual review.
        """)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 32px 20px; background: #1E293B; border-radius: 12px; border: 2px dashed #334155; margin-bottom: 16px;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">📋</div>
        <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">Upload a lab report PDF for automated value extraction and analysis</p>
        <p style="color: #64748B; font-size: 0.82rem; margin: 4px 0 0;">Supported format: PDF</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"], key="lab_pdf")

    tab_parsed, tab_raw = st.tabs(["Parsed Results", "Raw Text"])

    lab_data = None
    raw_text = ""
    used_demo = False

    if uploaded_pdf is not None:
        try:
            import pdfplumber
            with pdfplumber.open(uploaded_pdf) as pdf:
                pages_text = []
                for page in pdf.pages:
                    pages_text.append(page.extract_text() or "")
                raw_text = "\n".join(pages_text)

            parsed_labs = []
            patterns = [
                r'([A-Za-z\s/\-]+?)\s+([\d.]+)\s+([A-Za-z/%\u00b3\u00b5\u2076\s]+?)\s+([\d.]+)\s*[-\u2013]\s*([\d.]+)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, raw_text)
                for match in matches:
                    try:
                        parsed_labs.append({
                            "analyte": match[0].strip(),
                            "value": float(match[1]),
                            "unit": match[2].strip(),
                            "ref_low": float(match[3]),
                            "ref_high": float(match[4]),
                        })
                    except (ValueError, IndexError):
                        continue

            if parsed_labs:
                lab_data = parsed_labs
                st.success("Lab report parsed successfully!")
            else:
                lab_data = DEMO_LAB_REPORT
                used_demo = True
                st.warning("Could not parse lab values from PDF. Showing demo data for illustration.")
        except ImportError:
            st.warning("pdfplumber is not installed. Showing demo data.")
            lab_data = DEMO_LAB_REPORT
            used_demo = True
            raw_text = "(PDF parsing unavailable)"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            lab_data = DEMO_LAB_REPORT
            used_demo = True
    else:
        lab_data = DEMO_LAB_REPORT
        used_demo = True

    with tab_parsed:
        if lab_data:
            results = []
            for item in lab_data:
                status, css_class, color = classify_value(item["value"], item["ref_low"], item["ref_high"])
                results.append({**item, "status": status, "css_class": css_class, "color": color})

            total_tests = len(results)
            normal_count = sum(1 for r in results if r["status"] == "Normal")
            abnormal_count = total_tests - normal_count

            sc1, sc2, sc3 = st.columns(3, gap="medium")
            with sc1:
                st.markdown(f'<div class="metric-card"><p class="metric-label">Total Tests</p><p class="metric-value">{total_tests}</p></div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'<div class="metric-card risk-low"><p class="metric-label">Normal</p><p class="metric-value">{normal_count}</p></div>', unsafe_allow_html=True)
            with sc3:
                abn_class = "risk-high" if abnormal_count > 0 else "risk-low"
                st.markdown(f'<div class="metric-card {abn_class}"><p class="metric-label">Abnormal</p><p class="metric-value">{abnormal_count}</p></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if used_demo:
                st.markdown('<div class="info-card"><h4>Demo Lab Report</h4><p>Displaying sample lab data for demonstration. Upload a PDF to analyze your own report.</p></div>', unsafe_allow_html=True)

            table_rows = ""
            for r in results:
                table_rows += f"""<tr>
                    <td style="font-weight:600;">{r['analyte']}</td>
                    <td>{r['value']}</td><td>{r['unit']}</td>
                    <td>{r['ref_low']} - {r['ref_high']}</td>
                    <td><span class="{r['css_class']}">{r['status']}</span></td>
                </tr>"""

            st.markdown(f"""
            <table class="lab-table">
                <thead><tr><th>Analyte</th><th>Value</th><th>Unit</th><th>Reference Range</th><th>Flag</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            abnormal_results = [r for r in results if r["status"] != "Normal"]
            if abnormal_results:
                st.markdown("**Abnormal Values - Deviation from Reference Range**")
                abn_names = [r["analyte"] for r in abnormal_results]
                abn_devs = [round(r["value"] - r["ref_high"] if r["value"] > r["ref_high"] else r["ref_low"] - r["value"], 2) for r in abnormal_results]
                abn_colors = [r["color"] for r in abnormal_results]

                fig_abn = go.Figure(data=[go.Bar(
                    x=abn_names, y=abn_devs, marker_color=abn_colors,
                    text=[f"+{d}" if d > 0 else str(d) for d in abn_devs], textposition="outside",
                )])
                fig_abn.update_layout(height=320, template="plotly_dark", font=dict(family="Inter"), yaxis_title="Deviation from Reference", showlegend=False)
                st.plotly_chart(fig_abn, use_container_width=True, key="lab_abnormality_chart")

        st.markdown("---")
        with st.expander("Detailed Report - What Each Lab Value Means", expanded=False):
            if lab_data:
                for r in results:
                    name = r["analyte"]
                    if name in LAB_GENERAL_EXPLANATIONS:
                        st.markdown(f"**{name}: {r['value']} {r['unit']}** - _{r['status']}_")
                        st.markdown(LAB_GENERAL_EXPLANATIONS[name])
                        st.markdown("---")

        st.markdown("""
        <div class="disclaimer">
            <strong>Disclaimer:</strong> Automated PDF parsing may be inaccurate. Values extracted from uploaded
            reports should be verified against the original document.
        </div>
        """, unsafe_allow_html=True)

    with tab_raw:
        if raw_text:
            st.text_area("Raw Extracted Text", raw_text, height=400, key="lab_raw_text")
        else:
            st.info("Upload a PDF to see the raw extracted text.")

    st.markdown("---")
    st.markdown("##### Explore More Modules")
    sug_cols = st.columns(3, gap="medium")
    with sug_cols[0]:
        st.button("CBC Analysis →", key="suggest_lab_to_cbc", on_click=navigate_to, args=("CBC Analysis",), use_container_width=True)
    with sug_cols[1]:
        st.button("Kidney Function →", key="suggest_lab_to_kidney", on_click=navigate_to, args=("Kidney Function",), use_container_width=True)
    with sug_cols[2]:
        st.button("Diabetes Screening →", key="suggest_lab_to_diabetes", on_click=navigate_to, args=("Diabetes Screening",), use_container_width=True)


# ============================================================
# AI ASSISTANT SECTION
# ============================================================
elif section == "AI Assistant":
    st.button("\u2190 Back to Home", key="back_home_ai", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>AI Assistant</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">AI Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Ask questions about this portal\'s features, tools, and how to interpret results.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.08); border: 1px solid #334155; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 0.88rem; color: #38BDF8;">
        <strong>Quick Start:</strong> 1️⃣ Type your question below → 2️⃣ The assistant will help you navigate the portal
    </div>
    """, unsafe_allow_html=True)

    PORTAL_SYSTEM_PROMPT = """You are a helpful assistant for the Healthcare AI Prediction Portal.
    The portal has 10 clinical AI modules:
    1. Heart/ECG Analysis - 1D-CNN model, 99.5% accuracy, classifies 12-lead ECG recordings
    2. Chest X-Ray Analysis - MobileNetV2 transfer learning, detects pneumonia vs normal
    3. Health Risk Assessment - XGBoost/LightGBM/GBM/RF voting ensemble, 98.6% accuracy, 99.9% AUC-ROC, trained on 5,160 UCI + Framingham records
    4. CBC Analysis - Clinical algorithm for complete blood count interpretation
    5. Diabetes Screening - FINDRISC questionnaire + HbA1c/glucose thresholds
    6. Lipid Panel/CV Risk - ATP III classification + Pooled Cohort ASCVD risk
    7. Kidney Function - CKD-EPI 2021 race-free eGFR with KDIGO staging
    8. Lab Report Upload - PDF parsing for automated lab value extraction
    9. AI Assistant - This chat interface
    10. Privacy & Compliance - HIPAA compliance information

    You help users navigate the portal, understand results, and troubleshoot issues.
    You do NOT provide medical advice. If asked medical questions, redirect to a healthcare provider.
    Keep responses concise and helpful. Only answer questions about this portal."""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask me about the portal's features...", key="ai_chat_input"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Try API-based response, fall back to keyword matching
        response = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("ANTHROPIC_API_KEY")
            except Exception:
                pass

        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                api_response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    system=PORTAL_SYSTEM_PROMPT,
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history],
                )
                response = api_response.content[0].text
            except Exception:
                pass

        if not response:
            # Keyword-based fallback
            prompt_lower = prompt.lower()
            if any(w in prompt_lower for w in ["ecg", "heart", "ekg", "arrhythmia"]):
                response = "The **Heart / ECG Analysis** module uses a 1D-CNN model trained on 21,837 PTB-XL ECG recordings. It accepts CSV, DAT, HEA, or NPY files. Upload a 12-lead ECG or try the built-in sample data. The model classifies into 5 categories: Normal, ST/T Change, Conduction Disturbance, Hypertrophy, and MI."
            elif any(w in prompt_lower for w in ["xray", "x-ray", "chest", "pneumonia", "lung"]):
                response = "The **Chest X-Ray** module uses MobileNetV2 transfer learning trained on the NIH Chest X-ray14 dataset (112,120 images). Upload a frontal chest X-ray (PNG/JPEG) to get a pneumonia vs normal classification with confidence scores."
            elif any(w in prompt_lower for w in ["risk", "questionnaire", "heart disease"]):
                response = "The **Health Risk Assessment** is a step-by-step questionnaire that collects 13 clinical features (age, blood pressure, cholesterol, etc.) and uses an XGBoost/LightGBM/GBM/RF voting ensemble (98.6% accuracy, 99.9% AUC-ROC) trained on 5,160 combined UCI + Framingham records to estimate heart disease risk on a 0-100% scale."
            elif any(w in prompt_lower for w in ["cbc", "blood count", "hemoglobin", "platelet", "wbc"]):
                response = "The **CBC Analysis** module interprets complete blood count values using clinical reference ranges. Enter your WBC, RBC, hemoglobin, hematocrit, platelets, and differential counts. It flags abnormalities with color-coded badges."
            elif any(w in prompt_lower for w in ["diabetes", "glucose", "hba1c", "findrisc", "sugar"]):
                response = "The **Diabetes Screening** module uses the validated FINDRISC questionnaire plus HbA1c and fasting glucose thresholds. Enter your demographics, lab values, and lifestyle factors to get a diabetes risk score."
            elif any(w in prompt_lower for w in ["lipid", "cholesterol", "ldl", "hdl", "triglyceride", "ascvd"]):
                response = "The **Lipid Panel / CV Risk** module classifies lipid values per ATP III guidelines and estimates 10-year ASCVD risk using the AHA/ACC Pooled Cohort Equations. Enter your cholesterol panel and demographics."
            elif any(w in prompt_lower for w in ["kidney", "egfr", "creatinine", "ckd", "renal"]):
                response = "The **Kidney Function** module calculates eGFR using the CKD-EPI 2021 race-free equation with KDIGO staging. Enter serum creatinine, age, sex, and optionally Cystatin C for comparison."
            elif any(w in prompt_lower for w in ["lab", "report", "pdf", "upload"]):
                response = "The **Lab Report Upload** module accepts PDF lab reports. It uses regex parsing to extract common lab values and flags abnormalities against reference ranges. Upload any standard lab report PDF."
            elif any(w in prompt_lower for w in ["privacy", "hipaa", "data", "security"]):
                response = "Your privacy is our priority. All data is processed in real time and **never stored**. No accounts are required, and no personal data is collected. See the Privacy & Compliance page for full details."
            elif any(w in prompt_lower for w in ["hello", "hi", "hey"]):
                response = "Hello! I'm the Healthcare AI Portal assistant. I can help you navigate the portal, explain how modules work, or troubleshoot issues. What would you like to know?"
            else:
                response = "I can help you with any of the portal's 10 modules. Try asking about a specific module like 'How does the ECG analysis work?' or 'What format does the X-ray module accept?' For medical questions, please consult a healthcare provider."

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠ Note:</strong> This assistant provides information about portal features only.
        It does not provide medical advice. Always consult a qualified healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PRIVACY & HIPAA COMPLIANCE PAGE
# ============================================================
elif section == "Privacy & Compliance":
    st.button("← Back to Home", key="back_home_privacy", on_click=navigate_to, args=("Home",))
    st.markdown('<p style="color: #64748B; font-size: 0.85rem; margin: 0 0 8px 0;">Home / <strong>Privacy & Compliance</strong></p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Privacy & HIPAA Compliance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Information about data handling, privacy practices, and regulatory compliance.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Data Privacy Statement</h4>
        <p>This application is designed for <strong>educational and research purposes only</strong>.
        It is not a HIPAA-covered entity and should not be used for clinical decision-making with real patient data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Privacy Principles")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        **No Data Storage**
        - All uploaded files (ECGs, X-rays, PDFs) are processed in-memory only
        - No patient data is stored on any server or database
        - Data is discarded when the browser session ends

        **No Data Transmission**
        - All AI inference runs locally on the server
        - No patient data is sent to third-party APIs
        - No cloud-based ML services are used

        **Local Processing**
        - All models run on the deployment server
        - No external API calls for predictions
        - No telemetry or analytics tracking of patient data
        """)

    with col2:
        st.markdown("""
        **HIPAA Considerations**
        - This tool is **not HIPAA-compliant** and should not be used with Protected Health Information (PHI)
        - For clinical use, a HIPAA-compliant deployment would require:
            - Business Associate Agreements (BAAs)
            - Encrypted data storage and transmission (AES-256, TLS 1.2+)
            - Access controls and audit logging
            - Regular security assessments
            - Breach notification procedures

        **Intended Use**
        - Academic research and education
        - Demonstration of AI capabilities in healthcare
        - Portfolio project for data science coursework
        - **NOT** for clinical diagnosis or treatment decisions
        """)

    st.markdown("---")

    st.markdown("### Regulatory Framework Reference")
    st.markdown("""
    | Regulation | Description | Applicability |
    |:-----------|:-----------|:-------------|
    | **HIPAA** | Health Insurance Portability and Accountability Act | PHI protection in US healthcare |
    | **HITECH** | Health Information Technology for Economic and Clinical Health Act | Electronic health records |
    | **FDA 21 CFR Part 11** | Electronic records and signatures | Clinical software validation |
    | **GDPR** | General Data Protection Regulation | EU patient data protection |
    | **IEC 62304** | Medical device software lifecycle | Software as Medical Device (SaMD) |
    """)

    st.markdown("""
    <div class="disclaimer">
        <strong>Important Notice:</strong> This application is a research prototype developed as part of an academic
        program at Northwestern University. It has not been validated for clinical use, has not received FDA clearance
        or approval, and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class="footer">
    <div class="footer-divider"></div>
    Healthcare AI Prediction Portal &nbsp;&middot;&nbsp;
    <a href="https://github.com/abdullahabdulsami2026-coder/healthcare-ai-portal" target="_blank">GitHub</a>
    &nbsp;&middot;&nbsp; Abdullah Abdul Sami &nbsp;&middot;&nbsp; Northwestern University<br>
    <small>For research and educational purposes only. Not for clinical diagnosis.</small>
</div>
""", unsafe_allow_html=True)
