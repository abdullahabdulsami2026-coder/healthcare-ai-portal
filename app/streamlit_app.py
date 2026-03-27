"""
Healthcare AI Prediction Portal
================================
Multi-section dashboard for medical data analysis and prediction.

Sections:
1. Home — Overview and feature cards
2. Heart / ECG Analysis — Upload ECG, get arrhythmia classification
3. Chest X-Ray Analysis — Upload X-ray, get pneumonia/disease prediction
4. Health Risk Assessment — Input vitals, get heart disease risk score
5. CBC Analysis — Complete blood count analysis with clinical interpretation
6. Diabetes Screening — HbA1c, fasting glucose, and FINDRISC score
7. Lipid Panel / CV Risk — Lipid classification and 10-year ASCVD risk
8. Kidney Function — CKD-EPI eGFR, KDIGO staging, albuminuria
9. Lab Report Upload — Parse and analyze lab report PDFs

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

from utils.clinical_calculators import (
    classify_value, CBC_RANGES, interpret_cbc,
    calculate_findrisc, classify_hba1c, classify_fasting_glucose,
    classify_lipid, LIPID_CLASSES, calculate_ascvd_risk,
    ckd_epi_creatinine, ckd_epi_cystatin, stage_ckd, stage_albuminuria,
    DEMO_LAB_REPORT,
)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Healthcare AI Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — polished, modern healthcare UI
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main { background: linear-gradient(180deg, #f0f4f8 0%, #e8eef3 100%); }

    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* ── Hero Banner ────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
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
        background: radial-gradient(circle, rgba(46,134,193,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: rgba(255,255,255,0.75);
        font-size: 1.05rem;
        margin: 0;
        max-width: 600px;
        line-height: 1.6;
    }

    /* ── Stat Pill Row ──────────────────────────── */
    .stat-row {
        display: flex;
        gap: 12px;
        margin-top: 24px;
        flex-wrap: wrap;
    }
    .stat-pill {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 40px;
        padding: 8px 20px;
        color: white;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stat-pill strong { font-weight: 700; color: #5dade2; }

    /* ── Feature Cards ──────────────────────────── */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 28px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border-color: rgba(26,82,118,0.15);
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
    .icon-ecg { background: linear-gradient(135deg, #fce4ec, #f8bbd0); }
    .icon-xray { background: linear-gradient(135deg, #e3f2fd, #bbdefb); }
    .icon-risk { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); }
    .icon-cbc { background: linear-gradient(135deg, #f3e5f5, #ce93d8); }
    .icon-diabetes { background: linear-gradient(135deg, #fff3e0, #ffcc80); }
    .icon-lipid { background: linear-gradient(135deg, #e0f7fa, #80deea); }
    .icon-kidney { background: linear-gradient(135deg, #fce4ec, #ef9a9a); }
    .icon-lab { background: linear-gradient(135deg, #e8eaf6, #9fa8da); }

    .feature-card h3 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a2332;
        margin: 0 0 8px 0;
    }
    .feature-card p {
        color: #6b7b8d;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 0;
    }
    .feature-tag {
        display: inline-block;
        background: #f0f4f8;
        color: #4a6274;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 14px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ── Metric Cards ───────────────────────────── */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 22px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);
        border-left: 4px solid #2c5364;
        margin-bottom: 12px;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1a2332;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8899a6;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .risk-high { border-left-color: #e74c3c !important; }
    .risk-high .metric-value { color: #c0392b; }
    .risk-medium { border-left-color: #f39c12 !important; }
    .risk-medium .metric-value { color: #e67e22; }
    .risk-low { border-left-color: #27ae60 !important; }
    .risk-low .metric-value { color: #229954; }

    /* ── Section Headers ────────────────────────── */
    .section-header {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1a2332;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .section-sub {
        color: #6b7b8d;
        font-size: 0.95rem;
        margin-bottom: 28px;
        line-height: 1.5;
    }

    /* ── Result Box ─────────────────────────────── */
    .result-box {
        background: linear-gradient(135deg, #0f2027, #2c5364);
        color: white;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 30px rgba(15,32,39,0.2);
    }
    .result-box h2 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .result-box p {
        color: rgba(255,255,255,0.7);
        margin: 6px 0 0;
        font-size: 0.95rem;
    }

    /* ── Info Cards (datasets, about) ───────────── */
    .info-card {
        background: white;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .info-card h4 {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1a2332;
        margin: 0 0 6px 0;
    }
    .info-card p {
        font-size: 0.85rem;
        color: #6b7b8d;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Upload Zone ────────────────────────────── */
    .upload-zone {
        background: white;
        border: 2px dashed #d0d9e1;
        border-radius: 16px;
        padding: 40px 24px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .upload-zone:hover {
        border-color: #2c5364;
        background: #f8fafb;
    }
    .upload-icon { font-size: 2.5rem; margin-bottom: 12px; }
    .upload-text { color: #6b7b8d; font-size: 0.9rem; }

    /* ── Sidebar Styling ────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #1a3a47 100%);
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 500;
        padding: 8px 4px;
        border-radius: 8px;
        transition: background 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: rgba(255,255,255,0.12);
    }

    /* ── Button Styling ─────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0f2027, #2c5364) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(15,32,39,0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Tab Styling ────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: white;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }

    /* ── Footer ─────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 24px 0 12px;
        color: #8899a6;
        font-size: 0.78rem;
        letter-spacing: 0.01em;
    }
    .footer a { color: #2c5364; text-decoration: none; font-weight: 600; }
    .footer-divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #d0d9e1, transparent);
        margin: 0 auto 16px;
    }

    /* ── Flag Badges (lab values) ────────────────── */
    .flag-critical {
        background: #fadbd8;
        color: #922b21;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .flag-high {
        background: #fdebd0;
        color: #e67e22;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .flag-low {
        background: #fdebd0;
        color: #e67e22;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .flag-normal {
        background: #d5f5e3;
        color: #27ae60;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ── Lab Table ───────────────────────────────── */
    .lab-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .lab-table th {
        background: #f0f4f8;
        padding: 10px 14px;
        text-align: left;
        font-weight: 600;
        color: #4a6274;
        border-bottom: 2px solid #e0e6ec;
    }
    .lab-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #eef1f5;
    }

    /* ── Disclaimer ──────────────────────────────── */
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.82rem;
        color: #856404;
        margin: 16px 0;
    }

    /* ── CKD / KDIGO Grid ────────────────────────── */
    .ckd-grid {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .ckd-grid th {
        background: #f0f4f8;
        padding: 10px 14px;
        text-align: center;
        font-weight: 600;
        color: #4a6274;
        border-bottom: 2px solid #e0e6ec;
    }
    .ckd-grid td {
        padding: 8px 14px;
        border-bottom: 1px solid #eef1f5;
        text-align: center;
    }
    .ckd-cell {
        padding: 8px;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* ── Hide Streamlit defaults ────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Navigation State
# ============================================================
NAV_OPTIONS = [
    "Home", "Heart / ECG", "Chest X-Ray", "Health Risk Assessment",
    "CBC Analysis", "Diabetes Screening", "Lipid Panel / CV Risk",
    "Kidney Function", "Lab Report Upload",
]


def navigate_to(section_name):
    """Set navigation to a specific section by updating the radio key directly."""
    if section_name in NAV_OPTIONS:
        st.session_state.nav_radio = section_name


# ============================================================
# Sidebar Navigation
# ============================================================
with st.sidebar:
    st.markdown("### Healthcare AI")
    st.markdown("**Prediction Portal**")
    st.markdown("---")

    section = st.radio(
        "Navigation",
        NAV_OPTIONS,
        key="nav_radio",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model status indicators
    st.markdown("##### Model Status")
    heart_ok = os.path.exists(os.path.join(MODELS_DIR, "heart_risk.joblib"))
    ecg_ok = os.path.exists(os.path.join(MODELS_DIR, "ecg_classifier.h5"))
    xray_ok = os.path.exists(os.path.join(MODELS_DIR, "xray_classifier.h5"))

    st.markdown(f"{'🟢' if heart_ok else '🟡'} Heart Risk Model")
    st.markdown(f"{'🟢' if ecg_ok else '🟡'} ECG Classifier")
    st.markdown(f"{'🟢' if xray_ok else '🟡'} X-Ray Classifier")
    st.markdown("🟢 CBC Analysis — Algorithm")
    st.markdown("🟢 Diabetes Screening — Algorithm")
    st.markdown("🟢 Lipid Panel / CV Risk — Algorithm")
    st.markdown("🟢 Kidney Function — Algorithm")
    st.markdown("🟢 Lab Report Upload — Algorithm")

    st.markdown("---")
    st.markdown("##### Built by")
    st.markdown("**Abdullah Abdul Sami**")
    st.caption("MS Data Science (AI)")
    st.caption("Northwestern University")
    st.markdown("---")
    st.caption("For research and educational purposes only. Not for clinical diagnosis.")


# ============================================================
# HOME SECTION
# ============================================================
if section == "Home":
    # Hero banner
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
            <span class="stat-pill"><strong>9</strong>&nbsp; Clinical Modules</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards — clickable navigation
    CARD_DATA = [
        ("Heart / ECG", "icon-ecg", "❤️", "Heart / ECG Analysis",
         "Upload 12-lead ECG recordings for real-time arrhythmia classification. Get heart rate, HRV metrics, and diagnostic probabilities.", "1D CNN Model"),
        ("Chest X-Ray", "icon-xray", "🫁", "Chest X-Ray Analysis",
         "Upload frontal chest X-ray images for pneumonia detection and multi-label disease classification.", "Transfer Learning"),
        ("Health Risk Assessment", "icon-risk", "📊", "Health Risk Assessment",
         "Enter patient vitals and clinical data to generate a heart disease risk score with interactive charts.", "Random Forest"),
        ("CBC Analysis", "icon-cbc", "🩸", "CBC Analysis",
         "Enter complete blood count values for automated classification, WBC differential visualization, and clinical interpretation.", "Clinical Algorithm"),
        ("Diabetes Screening", "icon-diabetes", "🍩", "Diabetes Screening",
         "Comprehensive diabetes risk assessment using HbA1c, fasting glucose, and the validated FINDRISC questionnaire.", "FINDRISC"),
        ("Lipid Panel / CV Risk", "icon-lipid", "🫀", "Lipid Panel / CV Risk",
         "Lipid classification per ATP III guidelines with 10-year ASCVD risk estimation using Pooled Cohort Equations.", "Pooled Cohort Equations"),
        ("Kidney Function", "icon-kidney", "🫘", "Kidney Function",
         "CKD-EPI 2021 race-free eGFR calculation with KDIGO staging, albuminuria assessment, and risk matrix.", "CKD-EPI 2021"),
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
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-icon {icon_cls}">{emoji}</div>
                        <h3>{title}</h3>
                        <p>{desc}</p>
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
            else:
                with col:
                    st.markdown("""
                    <div class="info-card" style="height:100%;">
                        <h4>More Coming Soon</h4>
                        <p>Thyroid function, liver panel, and coagulation studies.</p>
                    </div>
                    """, unsafe_allow_html=True)
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
            Random Forest classifier (200 trees) achieves 83% accuracy.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# HEART / ECG SECTION
# ============================================================
elif section == "Heart / ECG":
    st.markdown('<p class="section-header">Heart / ECG Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload a 12-lead ECG recording or explore the demo analysis with simulated data.</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Upload ECG", "Demo Analysis"])

    # --- Upload Tab ---
    with tab1:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">📤</div>
            <div class="upload-text">Drag and drop a 12-lead ECG file below<br>
            <small>Supported: CSV, NumPy (.npy), WFDB (.dat/.hea)</small></div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose ECG file", type=["csv", "dat", "hea", "npy"],
            help="CSV (columns = leads), NumPy arrays, or WFDB format",
            label_visibility="collapsed",
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
                    st.success(f"Loaded ECG signal: {ecg_data.shape[0]} samples x {ecg_data.shape[1] if ecg_data.ndim > 1 else 1} leads")

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
                        template="plotly_white",
                        font=dict(family="Inter"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    try:
                        from utils.model_utils import load_model, predict_with_confidence
                        from utils.ecg_utils import prepare_ecg_for_model, DIAGNOSTIC_CLASSES

                        model = load_model("ecg_classifier", MODELS_DIR)
                        processed = prepare_ecg_for_model(ecg_data)
                        result = predict_with_confidence(
                            model, processed,
                            class_names=list(DIAGNOSTIC_CLASSES.values())
                        )

                        st.markdown(f"""
                        <div class="result-box">
                            <h2>{result['predicted_label']}</h2>
                            <p>Confidence: {result['confidence']}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                        probs_df = pd.DataFrame(
                            list(result["all_probabilities"].items()),
                            columns=["Condition", "Probability (%)"]
                        )
                        fig_probs = px.bar(
                            probs_df, x="Probability (%)", y="Condition",
                            orientation="h", color="Probability (%)",
                            color_continuous_scale=["#e8eef3", "#2c5364"],
                        )
                        fig_probs.update_layout(
                            height=300, template="plotly_white",
                            font=dict(family="Inter"),
                        )
                        st.plotly_chart(fig_probs, use_container_width=True)

                    except FileNotFoundError:
                        st.info("ECG model not available. Running in demo mode.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

            except Exception as e:
                st.error(f"Error loading file: {e}")

    # --- Demo Tab ---
    with tab2:
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

        # Metric cards row
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

        # ECG trace
        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(
            x=t[:2000], y=ecg_sim[:2000],
            mode="lines", name="Lead II",
            line=dict(color="#2c5364", width=1.5)
        ))
        fig_demo.update_layout(
            title=dict(text="12-Lead ECG — Lead II", font=dict(size=16, family="Inter")),
            xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
            height=380, template="plotly_white",
            font=dict(family="Inter"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_demo, use_container_width=True)

        # Probability chart
        demo_probs = pd.DataFrame({
            "Condition": ["Normal ECG", "ST/T Change", "Conduction Dist.", "Hypertrophy", "MI"],
            "Probability (%)": [89.2, 5.1, 3.2, 1.8, 0.7]
        })
        fig_p = px.bar(
            demo_probs, x="Probability (%)", y="Condition",
            orientation="h", color="Probability (%)",
            color_continuous_scale=["#e8eef3", "#2c5364"],
        )
        fig_p.update_layout(
            height=260, template="plotly_white",
            font=dict(family="Inter"),
            showlegend=False,
        )
        st.plotly_chart(fig_p, use_container_width=True)


# ============================================================
# CHEST X-RAY SECTION
# ============================================================
elif section == "Chest X-Ray":
    st.markdown('<p class="section-header">Chest X-Ray Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload a frontal chest X-ray image (PA or AP view) for AI-powered pneumonia detection.</p>', unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded_xray = st.file_uploader(
            "Choose X-ray image", type=["png", "jpg", "jpeg", "dcm"],
            help="Supported: PNG, JPEG. Frontal PA/AP view recommended.",
        )

        if uploaded_xray is not None:
            from PIL import Image
            img = Image.open(uploaded_xray).convert("RGB")
            st.image(img, caption="Uploaded X-ray", use_container_width=True)
        else:
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-icon">🫁</div>
                <div class="upload-text">Upload a chest X-ray image<br>
                <small>PNG, JPEG, or DICOM format</small></div>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        if uploaded_xray is not None:
            try:
                from utils.model_utils import load_model, predict_with_confidence
                from utils.xray_utils import (
                    load_and_preprocess_xray, prepare_xray_for_model,
                    PNEUMONIA_CLASSES,
                )

                img_array = load_and_preprocess_xray(uploaded_xray)
                img_batch = prepare_xray_for_model(img_array)

                model = load_model("xray_classifier", MODELS_DIR)
                result = predict_with_confidence(model, img_batch, class_names=PNEUMONIA_CLASSES)

                risk_class = "risk-high" if result["predicted_label"] == "Pneumonia" else "risk-low"
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <p class="metric-label">Prediction</p>
                    <p class="metric-value">{result['predicted_label']}</p>
                    <p style="color:#6b7b8d; font-size:0.9rem; margin-top:4px;">
                        Confidence: {result['confidence']}%</p>
                </div>
                """, unsafe_allow_html=True)

                probs_df = pd.DataFrame(
                    list(result["all_probabilities"].items()),
                    columns=["Class", "Probability (%)"]
                )
                fig = px.pie(probs_df, values="Probability (%)", names="Class",
                             color_discrete_sequence=["#27ae60", "#e74c3c"],
                             hole=0.4)
                fig.update_layout(
                    height=300, font=dict(family="Inter"),
                    margin=dict(t=20, b=20),
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

            except FileNotFoundError:
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
                             color_discrete_sequence=["#27ae60", "#e74c3c"],
                             hole=0.4)
                fig.update_layout(
                    height=300, font=dict(family="Inter"),
                    margin=dict(t=20, b=20),
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div class="info-card">
                <h4>How it works</h4>
                <p>1. Upload a frontal chest X-ray image (PNG or JPEG)<br>
                2. The AI model preprocesses and analyzes the image<br>
                3. Get pneumonia probability with confidence score<br>
                4. View the prediction breakdown chart</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4>Model Details</h4>
                <p>MobileNetV2 backbone with transfer learning from ImageNet.
                Fine-tuned on Kaggle Chest X-Ray Pneumonia dataset
                (5,863 labeled images).</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# HEALTH RISK ASSESSMENT SECTION
# ============================================================
elif section == "Health Risk Assessment":
    st.markdown('<p class="section-header">Health Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter patient clinical data to generate a heart disease risk prediction with interactive visualizations.</p>', unsafe_allow_html=True)

    # Input form in a clean container
    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", min_value=1, max_value=120, value=55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", [
                "Typical Angina", "Atypical Angina",
                "Non-Anginal Pain", "Asymptomatic"
            ])
            trestbps = st.number_input("Resting BP (mmHg)", 80, 220, 130)

        with col2:
            st.markdown("**Lab Values**")
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)
            fbs = st.selectbox("Fasting Blood Sugar > 120?", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG", [
                "Normal", "ST-T Abnormality", "LV Hypertrophy"
            ])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)

        with col3:
            st.markdown("**Cardiac Tests**")
            exang = st.selectbox("Exercise Angina?", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
            ca = st.number_input("Major Vessels (0-3)", 0, 3, 0)
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    st.markdown("")  # spacer

    if st.button("Predict Risk", type="primary", use_container_width=True):
        # Encode inputs
        sex_val = 1 if sex == "Male" else 0
        cp_val = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
        fbs_val = 1 if fbs == "Yes" else 0
        restecg_val = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
        exang_val = 1 if exang == "Yes" else 0
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

        features = np.array([[
            age, sex_val, cp_val, trestbps, chol, fbs_val,
            restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val
        ]])

        try:
            from utils.model_utils import load_model, predict_with_confidence
            model = load_model("heart_risk", MODELS_DIR)
            result = predict_with_confidence(model, features, class_names=["Low Risk", "High Risk"])
            risk_score = result["all_probabilities"].get("High Risk", 0)
            predicted = result["predicted_label"]
        except FileNotFoundError:
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

            st.info("Using rule-based estimate. Train the model for ML predictions.")

        # ── Results ──
        st.markdown("---")

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

        # Charts side by side
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
            fig_gauge.update_layout(
                height=320,
                margin=dict(t=60, b=20, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_r:
            categories = ["Age", "BP", "Cholesterol", "Heart Rate", "ST Depression", "Vessels"]
            values = [
                age / 120 * 100,
                trestbps / 220 * 100,
                chol / 600 * 100,
                thalach / 220 * 100,
                oldpeak / 6 * 100,
                ca / 3 * 100,
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
                height=320,
                margin=dict(t=60, b=20, l=60, r=60),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ============================================================
# CBC ANALYSIS SECTION
# ============================================================
elif section == "CBC Analysis":
    st.markdown('<p class="section-header">CBC Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter complete blood count values for automated classification, differential visualization, and clinical interpretation.</p>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**Patient Info & Basic CBC**")
            cbc_sex = st.selectbox("Sex", ["Male", "Female"], key="cbc_sex")
            cbc_wbc = st.number_input("WBC (x10\u00b3/\u00b5L)", min_value=0.0, max_value=100.0, value=7.0, step=0.1, key="cbc_wbc")
            cbc_rbc = st.number_input("RBC (x10\u2076/\u00b5L)", min_value=0.0, max_value=15.0, value=4.7, step=0.1, key="cbc_rbc")
            cbc_hgb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=14.0, step=0.1, key="cbc_hgb")
            cbc_hct = st.number_input("Hematocrit (%)", min_value=0.0, max_value=80.0, value=42.0, step=0.1, key="cbc_hct")

        with col2:
            st.markdown("**RBC Indices**")
            cbc_mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=150.0, value=88.0, step=0.1, key="cbc_mcv")
            cbc_mch = st.number_input("MCH (pg)", min_value=0.0, max_value=50.0, value=29.0, step=0.1, key="cbc_mch")
            cbc_mchc = st.number_input("MCHC (g/dL)", min_value=0.0, max_value=45.0, value=33.5, step=0.1, key="cbc_mchc")
            cbc_rdw = st.number_input("RDW (%)", min_value=0.0, max_value=30.0, value=13.0, step=0.1, key="cbc_rdw")
            cbc_plt = st.number_input("Platelets (x10\u00b3/\u00b5L)", min_value=0.0, max_value=2000.0, value=250.0, step=1.0, key="cbc_plt")

        with col3:
            st.markdown("**WBC Differential (%)**")
            cbc_neut = st.number_input("Neutrophils %", min_value=0.0, max_value=100.0, value=60.0, step=0.1, key="cbc_neut")
            cbc_lymph = st.number_input("Lymphocytes %", min_value=0.0, max_value=100.0, value=30.0, step=0.1, key="cbc_lymph")
            cbc_mono = st.number_input("Monocytes %", min_value=0.0, max_value=100.0, value=6.0, step=0.1, key="cbc_mono")
            cbc_eos = st.number_input("Eosinophils %", min_value=0.0, max_value=100.0, value=3.0, step=0.1, key="cbc_eos")
            cbc_baso = st.number_input("Basophils %", min_value=0.0, max_value=100.0, value=0.5, step=0.1, key="cbc_baso")

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

        st.markdown("---")

        # Metric cards for key values
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
            if status == "Normal":
                card_class = "risk-low"
            elif "Critical" in status:
                card_class = "risk-high"
            else:
                card_class = "risk-medium"

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
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Unit</th>
                    <th>Reference Range</th>
                    <th>Flag</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # WBC Differential donut chart
        col_donut, col_interp = st.columns(2, gap="medium")

        with col_donut:
            diff_labels = ["Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils"]
            diff_values = [cbc_neut, cbc_lymph, cbc_mono, cbc_eos, cbc_baso]
            diff_colors = ["#2c5364", "#5dade2", "#48c9b0", "#f4d03f", "#e74c3c"]

            fig_diff = go.Figure(data=[go.Pie(
                labels=diff_labels, values=diff_values,
                hole=0.4,
                marker=dict(colors=diff_colors),
                textinfo="label+percent",
                textfont=dict(family="Inter", size=12),
            )])
            fig_diff.update_layout(
                title=dict(text="WBC Differential", font=dict(size=16, family="Inter")),
                height=350,
                template="plotly_white",
                font=dict(family="Inter"),
                margin=dict(t=60, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_diff, use_container_width=True)

        with col_interp:
            findings = interpret_cbc(cbc_values, sex_key)
            st.markdown("**Clinical Interpretation**")
            for finding in findings:
                st.info(finding)

        st.markdown("""
        <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This CBC analysis is generated by an automated algorithm using
            standard reference ranges. It is intended for educational and screening purposes only. Always consult
            a qualified healthcare provider for clinical decision-making.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# DIABETES SCREENING SECTION
# ============================================================
elif section == "Diabetes Screening":
    st.markdown('<p class="section-header">Diabetes Screening</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Comprehensive diabetes risk assessment using HbA1c, fasting glucose, and the FINDRISC questionnaire.</p>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**Lab Values**")
            dm_hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1, key="dm_hba1c")
            dm_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=40, max_value=500, value=95, key="dm_glucose")

            st.markdown("**Demographics**")
            dm_age = st.number_input("Age", min_value=18, max_value=120, value=50, key="dm_age")
            dm_bmi = st.number_input("BMI (kg/m\u00b2)", min_value=10.0, max_value=60.0, value=25.0, step=0.1, key="dm_bmi")
            dm_waist = st.number_input("Waist Circumference (cm)", min_value=50, max_value=200, value=90, key="dm_waist")
            dm_sex = st.selectbox("Sex", ["Male", "Female"], key="dm_sex")

        with col2:
            st.markdown("**Family & History**")
            dm_family = st.selectbox("Family History of Diabetes", ["None", "One parent", "Both parents"], key="dm_family")
            dm_activity = st.selectbox("Physical Activity", ["Active", "Low"], key="dm_activity")
            dm_fruit = st.selectbox("Daily Fruit/Vegetable Intake", ["Yes", "No"], key="dm_fruit")

        with col3:
            st.markdown("**Medical History**")
            dm_bp_med = st.selectbox("BP Medication", ["No", "Yes"], key="dm_bp_med")
            dm_high_glucose = st.selectbox("History of High Blood Glucose", ["No", "Yes"], key="dm_high_glucose")

    st.markdown("")

    if st.button("Screen", type="primary", use_container_width=True, key="dm_screen"):
        # Classify lab values
        hba1c_label, hba1c_color = classify_hba1c(dm_hba1c)
        glucose_label, glucose_color = classify_fasting_glucose(dm_glucose)

        # Map inputs for FINDRISC
        family_map = {"None": "none", "One parent": "one_parent", "Both parents": "both_parents"}
        findrisc_score, findrisc_cat, findrisc_risk = calculate_findrisc(
            age=dm_age,
            bmi=dm_bmi,
            waist=dm_waist,
            sex=dm_sex.lower(),
            activity=dm_activity.lower(),
            fruit_veg=dm_fruit.lower(),
            bp_meds=dm_bp_med.lower(),
            high_glucose=dm_high_glucose.lower(),
            family_hx=family_map[dm_family],
        )

        st.markdown("---")

        # Metric cards
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
            if findrisc_score < 7:
                fr_card = "risk-low"
            elif findrisc_score <= 14:
                fr_card = "risk-medium"
            else:
                fr_card = "risk-high"
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
            # FINDRISC gauge chart
            fig_findrisc = go.Figure(go.Indicator(
                mode="gauge+number",
                value=findrisc_score,
                number={"font": {"size": 42, "family": "Inter", "color": "#1a2332"}},
                title={"text": f"FINDRISC Score — {findrisc_cat}", "font": {"size": 16, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 26], "tickwidth": 1, "tickcolor": "#d0d9e1"},
                    "bar": {"color": "#2c5364", "thickness": 0.75},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 7], "color": "#d5f5e3"},
                        {"range": [7, 12], "color": "#eafaf1"},
                        {"range": [12, 15], "color": "#fef9e7"},
                        {"range": [15, 20], "color": "#fdebd0"},
                        {"range": [20, 26], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "#e74c3c", "width": 3},
                        "thickness": 0.8,
                        "value": findrisc_score,
                    },
                },
            ))
            fig_findrisc.update_layout(
                height=320,
                margin=dict(t=60, b=20, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_findrisc, use_container_width=True)

        with col_bar:
            # Risk factor contribution bar chart
            # Recalculate individual contributions
            contributions = {}

            if dm_age < 45: contributions["Age"] = 0
            elif dm_age <= 54: contributions["Age"] = 2
            elif dm_age <= 64: contributions["Age"] = 3
            else: contributions["Age"] = 4

            if dm_bmi < 25: contributions["BMI"] = 0
            elif dm_bmi <= 30: contributions["BMI"] = 1
            else: contributions["BMI"] = 3

            if dm_sex.lower() == "male":
                if dm_waist < 94: contributions["Waist"] = 0
                elif dm_waist <= 102: contributions["Waist"] = 3
                else: contributions["Waist"] = 4
            else:
                if dm_waist < 80: contributions["Waist"] = 0
                elif dm_waist <= 88: contributions["Waist"] = 3
                else: contributions["Waist"] = 4

            contributions["Activity"] = 2 if dm_activity == "Low" else 0
            contributions["Diet"] = 1 if dm_fruit == "No" else 0
            contributions["BP Meds"] = 2 if dm_bp_med == "Yes" else 0
            contributions["High Glucose Hx"] = 5 if dm_high_glucose == "Yes" else 0

            fam_map_pts = {"None": 0, "One parent": 3, "Both parents": 5}
            contributions["Family Hx"] = fam_map_pts[dm_family]

            contrib_df = pd.DataFrame({
                "Factor": list(contributions.keys()),
                "Points": list(contributions.values()),
            })
            contrib_df = contrib_df.sort_values("Points", ascending=True)

            fig_contrib = px.bar(
                contrib_df, x="Points", y="Factor",
                orientation="h",
                color="Points",
                color_continuous_scale=["#d5f5e3", "#f39c12", "#e74c3c"],
            )
            fig_contrib.update_layout(
                title=dict(text="Risk Factor Contributions", font=dict(size=16, family="Inter")),
                height=320,
                template="plotly_white",
                font=dict(family="Inter"),
                showlegend=False,
                xaxis_title="Points",
                yaxis_title="",
                margin=dict(t=60, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_contrib, use_container_width=True)

        # Clinical interpretation
        st.markdown("**Clinical Interpretation**")
        interp_msgs = []
        if hba1c_label == "Diabetes":
            interp_msgs.append("HbA1c is in the diabetic range (>= 6.5%). Confirmatory testing and clinical evaluation recommended.")
        elif hba1c_label == "Prediabetes":
            interp_msgs.append("HbA1c indicates prediabetes (5.7-6.4%). Lifestyle modification and monitoring advised.")

        if glucose_label == "Diabetes":
            interp_msgs.append("Fasting glucose is in the diabetic range (>= 126 mg/dL). Repeat testing recommended for confirmation.")
        elif glucose_label == "Prediabetes":
            interp_msgs.append("Fasting glucose indicates impaired fasting glucose (100-125 mg/dL).")

        interp_msgs.append(f"FINDRISC score of {findrisc_score} indicates {findrisc_cat.lower()} risk with an estimated 10-year probability of developing type 2 diabetes of {findrisc_risk}.")

        if findrisc_score >= 15:
            interp_msgs.append("High FINDRISC score warrants oral glucose tolerance testing (OGTT) and close follow-up.")

        if not interp_msgs:
            interp_msgs.append("All screening parameters are within normal ranges.")

        for msg in interp_msgs:
            st.info(msg)

        st.markdown("""
        <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This diabetes screening tool uses ADA criteria for HbA1c/glucose
            classification and the validated FINDRISC questionnaire. It is intended for screening purposes only and
            does not replace clinical judgment or diagnostic testing.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# LIPID PANEL / CV RISK SECTION
# ============================================================
elif section == "Lipid Panel / CV Risk":
    st.markdown('<p class="section-header">Lipid Panel / CV Risk</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Lipid classification and 10-year ASCVD risk estimation using the Pooled Cohort Equations.</p>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**Lipid Panel**")
            lp_tc = st.number_input("Total Cholesterol (mg/dL)", min_value=50, max_value=500, value=200, key="lp_tc")
            lp_ldl = st.number_input("LDL (mg/dL)", min_value=20, max_value=400, value=120, key="lp_ldl")
            lp_hdl = st.number_input("HDL (mg/dL)", min_value=10, max_value=150, value=50, key="lp_hdl")
            lp_trig = st.number_input("Triglycerides (mg/dL)", min_value=20, max_value=2000, value=150, key="lp_trig")

        with col2:
            st.markdown("**Demographics**")
            lp_age = st.number_input("Age", min_value=20, max_value=120, value=55, key="lp_age")
            lp_sex = st.selectbox("Sex", ["Male", "Female"], key="lp_sex")
            lp_race = st.selectbox("Race", ["White", "African American", "Other"], key="lp_race")

        with col3:
            st.markdown("**Risk Factors**")
            lp_sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=130, key="lp_sbp")
            lp_bp_med = st.selectbox("On BP Medication", ["No", "Yes"], key="lp_bp_med")
            lp_smoker = st.selectbox("Current Smoker", ["No", "Yes"], key="lp_smoker")
            lp_diabetes = st.selectbox("Diabetes", ["No", "Yes"], key="lp_diabetes")

    st.markdown("")

    if st.button("Assess Risk", type="primary", use_container_width=True, key="lp_assess"):
        # Classify lipids
        tc_label, tc_color = classify_lipid("Total Cholesterol", lp_tc)
        ldl_label, ldl_color = classify_lipid("LDL", lp_ldl)
        hdl_label, hdl_color = classify_lipid("HDL", lp_hdl)
        trig_label, trig_color = classify_lipid("Triglycerides", lp_trig)

        # Derived values
        non_hdl = lp_tc - lp_hdl
        tc_hdl_ratio = round(lp_tc / lp_hdl, 2) if lp_hdl > 0 else 0
        ldl_hdl_ratio = round(lp_ldl / lp_hdl, 2) if lp_hdl > 0 else 0

        # ASCVD risk
        ascvd_pct, ascvd_cat, ascvd_color = calculate_ascvd_risk(
            age=lp_age,
            sex=lp_sex.lower(),
            race=lp_race,
            total_chol=lp_tc,
            hdl=lp_hdl,
            sbp=lp_sbp,
            bp_treated=(lp_bp_med == "Yes"),
            smoker=(lp_smoker == "Yes"),
            diabetes=(lp_diabetes == "Yes"),
        )

        st.markdown("---")

        # Metric cards
        mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")

        with mc1:
            if ascvd_pct is not None:
                ascvd_card = "risk-low" if ascvd_pct < 5 else ("risk-high" if ascvd_pct >= 20 else "risk-medium")
            else:
                ascvd_card = ""
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
            # ASCVD gauge
            gauge_val = ascvd_pct if ascvd_pct is not None else 0
            fig_ascvd = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                number={"suffix": "%", "font": {"size": 42, "family": "Inter", "color": "#1a2332"}},
                title={"text": f"10-Year ASCVD Risk — {ascvd_cat}", "font": {"size": 16, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 30], "tickwidth": 1, "tickcolor": "#d0d9e1"},
                    "bar": {"color": "#2c5364", "thickness": 0.75},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 5], "color": "#d5f5e3"},
                        {"range": [5, 7.5], "color": "#eafaf1"},
                        {"range": [7.5, 20], "color": "#fdebd0"},
                        {"range": [20, 30], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "#e74c3c", "width": 3},
                        "thickness": 0.8,
                        "value": gauge_val,
                    },
                },
            ))
            fig_ascvd.update_layout(
                height=320,
                margin=dict(t=60, b=20, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_ascvd, use_container_width=True)

        with col_bar:
            # Lipid values bar chart
            lipid_names = ["Total Cholesterol", "LDL", "HDL", "Triglycerides"]
            lipid_vals = [lp_tc, lp_ldl, lp_hdl, lp_trig]
            lipid_colors = [tc_color, ldl_color, hdl_color, trig_color]

            fig_lipid = go.Figure(data=[go.Bar(
                x=lipid_names, y=lipid_vals,
                marker_color=lipid_colors,
                text=[f"{v} mg/dL" for v in lipid_vals],
                textposition="outside",
                textfont=dict(family="Inter", size=12),
            )])
            fig_lipid.update_layout(
                title=dict(text="Lipid Panel Values", font=dict(size=16, family="Inter")),
                height=320,
                template="plotly_white",
                font=dict(family="Inter"),
                yaxis_title="mg/dL",
                xaxis_title="",
                margin=dict(t=60, b=20, l=20, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_lipid, use_container_width=True)

        # Lipid ratios
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

        # Clinical interpretation
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Clinical Interpretation**")
        interp = []
        if ascvd_pct is not None:
            interp.append(f"Estimated 10-year ASCVD risk is {ascvd_pct}% ({ascvd_cat} risk). " +
                          ("Statin therapy should be considered." if ascvd_pct >= 7.5 else "Continue risk factor management."))
        else:
            interp.append("ASCVD risk calculation requires age 40-79.")

        if ldl_label in ["High", "Very High"]:
            interp.append(f"LDL cholesterol is {ldl_label.lower()} at {lp_ldl} mg/dL. ACC/AHA guidelines recommend statin therapy evaluation.")
        if hdl_label == "Low (Risk Factor)":
            interp.append(f"HDL cholesterol is low at {lp_hdl} mg/dL, an independent cardiovascular risk factor.")
        if trig_label in ["High", "Very High"]:
            interp.append(f"Triglycerides are {trig_label.lower()} at {lp_trig} mg/dL. Evaluate for secondary causes and consider treatment.")

        for msg in interp:
            st.info(msg)

        st.markdown("""
        <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This cardiovascular risk assessment uses the ACC/AHA Pooled Cohort
            Equations (2013) and ATP III lipid classifications. It is intended for screening and educational purposes
            only. Clinical decisions should be made in consultation with a healthcare provider.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# KIDNEY FUNCTION SECTION
# ============================================================
elif section == "Kidney Function":
    st.markdown('<p class="section-header">Kidney Function</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">CKD-EPI 2021 race-free eGFR estimation with KDIGO staging and risk classification.</p>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**Required Inputs**")
            kf_cr = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1, key="kf_cr")
            kf_age = st.number_input("Age", min_value=18, max_value=120, value=55, key="kf_age")
            kf_sex = st.selectbox("Sex", ["Male", "Female"], key="kf_sex")

        with col2:
            st.markdown("**Albuminuria & BUN**")
            kf_uacr = st.number_input("UACR (mg/g)", min_value=0.0, max_value=5000.0, value=15.0, step=1.0, key="kf_uacr")
            kf_bun = st.number_input("BUN (mg/dL)", min_value=1.0, max_value=150.0, value=15.0, step=0.5, key="kf_bun")

        with col3:
            st.markdown("**Optional: Cystatin C**")
            kf_use_cysc = st.checkbox("Include Cystatin C", value=False, key="kf_use_cysc")
            kf_cysc = st.number_input("Cystatin C (mg/L)", min_value=0.1, max_value=10.0, value=0.9, step=0.1, key="kf_cysc", disabled=not kf_use_cysc)

    st.markdown("")

    if st.button("Calculate", type="primary", use_container_width=True, key="kf_calc"):
        sex_key = kf_sex.lower()

        # CKD-EPI creatinine-based eGFR
        egfr_cr = ckd_epi_creatinine(kf_cr, kf_age, sex_key)
        ckd_stage, ckd_desc, ckd_color = stage_ckd(egfr_cr)
        alb_stage, alb_desc, alb_color = stage_albuminuria(kf_uacr)

        # Optional cystatin C
        egfr_cysc = None
        if kf_use_cysc:
            egfr_cysc = ckd_epi_cystatin(kf_cysc, kf_age, sex_key)

        # BUN/Creatinine ratio
        bun_cr_ratio = round(kf_bun / kf_cr, 1) if kf_cr > 0 else 0

        st.markdown("---")

        # Metric cards
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
            # eGFR gauge chart with CKD stage color bands
            fig_egfr = go.Figure(go.Indicator(
                mode="gauge+number",
                value=egfr_cr,
                number={"font": {"size": 42, "family": "Inter", "color": "#1a2332"}},
                title={"text": f"eGFR — Stage {ckd_stage}", "font": {"size": 16, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 120], "tickwidth": 1, "tickcolor": "#d0d9e1"},
                    "bar": {"color": "#2c5364", "thickness": 0.75},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 15], "color": "#fadbd8"},      # G5
                        {"range": [15, 30], "color": "#f5b7b1"},     # G4
                        {"range": [30, 45], "color": "#fdebd0"},     # G3b
                        {"range": [45, 60], "color": "#fef9e7"},     # G3a
                        {"range": [60, 90], "color": "#eafaf1"},     # G2
                        {"range": [90, 120], "color": "#d5f5e3"},    # G1
                    ],
                    "threshold": {
                        "line": {"color": "#e74c3c", "width": 3},
                        "thickness": 0.8,
                        "value": egfr_cr,
                    },
                },
            ))
            fig_egfr.update_layout(
                height=320,
                margin=dict(t=60, b=20, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_egfr, use_container_width=True)

        with col_comp:
            if egfr_cysc is not None:
                # Comparison bars
                comp_df = pd.DataFrame({
                    "Method": ["Creatinine-based", "Cystatin C-based"],
                    "eGFR": [egfr_cr, egfr_cysc],
                })
                fig_comp = go.Figure(data=[go.Bar(
                    x=comp_df["Method"], y=comp_df["eGFR"],
                    marker_color=["#2c5364", "#5dade2"],
                    text=[f"{v} mL/min" for v in comp_df["eGFR"]],
                    textposition="outside",
                    textfont=dict(family="Inter", size=13),
                )])
                fig_comp.update_layout(
                    title=dict(text="eGFR Comparison", font=dict(size=16, family="Inter")),
                    height=320,
                    template="plotly_white",
                    font=dict(family="Inter"),
                    yaxis_title="eGFR (mL/min/1.73m\u00b2)",
                    xaxis_title="",
                    margin=dict(t=60, b=20, l=20, r=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.markdown("""
                <div class="info-card">
                    <h4>Cystatin C Comparison</h4>
                    <p>Enable Cystatin C input to see a comparison between creatinine-based
                    and cystatin C-based eGFR estimates. Cystatin C may be more accurate
                    in certain populations (elderly, extreme muscle mass, etc.).</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KDIGO Risk Matrix
        st.markdown("**KDIGO Risk Matrix — Prognosis of CKD by GFR and Albuminuria**")

        # Define the risk colors for the KDIGO matrix
        # Rows: G1, G2, G3a, G3b, G4, G5
        # Cols: A1, A2, A3
        kdigo_colors = [
            ["#d5f5e3", "#eafaf1", "#fdebd0"],  # G1
            ["#d5f5e3", "#eafaf1", "#fdebd0"],  # G2
            ["#eafaf1", "#fdebd0", "#fadbd8"],  # G3a
            ["#fdebd0", "#fadbd8", "#fadbd8"],  # G3b
            ["#fadbd8", "#fadbd8", "#f1948a"],  # G4
            ["#fadbd8", "#f1948a", "#f1948a"],  # G5
        ]
        kdigo_labels = [
            ["Low", "Moderate", "High"],
            ["Low", "Moderate", "High"],
            ["Moderate", "High", "Very High"],
            ["High", "Very High", "Very High"],
            ["Very High", "Very High", "Very High"],
            ["Very High", "Very High", "Very High"],
        ]
        gfr_stages = [
            ("G1", "\u226590"),
            ("G2", "60-89"),
            ("G3a", "45-59"),
            ("G3b", "30-44"),
            ("G4", "15-29"),
            ("G5", "<15"),
        ]
        alb_stages = [
            ("A1", "<30"),
            ("A2", "30-300"),
            ("A3", ">300"),
        ]

        # Determine patient's position
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

        st.markdown("<br>", unsafe_allow_html=True)

        # Clinical interpretation
        st.markdown("**Clinical Interpretation**")
        interp = []
        interp.append(f"eGFR (creatinine-based) is {egfr_cr} mL/min/1.73m\u00b2, corresponding to CKD stage {ckd_stage} ({ckd_desc}).")
        if egfr_cysc is not None:
            cysc_stage, cysc_desc, _ = stage_ckd(egfr_cysc)
            interp.append(f"eGFR (cystatin C-based) is {egfr_cysc} mL/min/1.73m\u00b2, corresponding to CKD stage {cysc_stage} ({cysc_desc}).")
            if abs(egfr_cr - egfr_cysc) > 15:
                interp.append("Significant discordance between creatinine and cystatin C eGFR. Consider factors affecting creatinine (muscle mass, diet) or cystatin C (thyroid dysfunction, corticosteroids).")

        interp.append(f"Albuminuria stage: {alb_stage} ({alb_desc}) with UACR of {kf_uacr} mg/g.")

        if bun_cr_ratio > 20:
            interp.append(f"Elevated BUN/Creatinine ratio ({bun_cr_ratio}) may suggest pre-renal azotemia, GI bleeding, or high protein intake.")
        elif bun_cr_ratio < 10:
            interp.append(f"Low BUN/Creatinine ratio ({bun_cr_ratio}) may suggest liver disease, malnutrition, or overhydration.")

        risk_label = kdigo_labels[patient_row][patient_col] if patient_row >= 0 and patient_col >= 0 else "Unknown"
        interp.append(f"KDIGO composite risk category: {risk_label}. " +
                      ("Referral to nephrology recommended." if risk_label in ["High", "Very High"] else "Routine monitoring appropriate."))

        for msg in interp:
            st.info(msg)

        st.markdown("""
        <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This kidney function assessment uses the CKD-EPI 2021 race-free
            equations and KDIGO 2012 staging guidelines. It is intended for screening and educational purposes only.
            Clinical decisions should be made in consultation with a nephrologist or qualified healthcare provider.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# LAB REPORT UPLOAD SECTION
# ============================================================
elif section == "Lab Report Upload":
    st.markdown('<p class="section-header">Lab Report Upload</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload a lab report PDF for automated parsing, or explore the demo report with color-coded analysis.</p>', unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"], key="lab_pdf")

    tab_parsed, tab_raw = st.tabs(["Parsed Results", "Raw Text"])

    # Try to parse PDF
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

            # Attempt regex extraction
            # Pattern: analyte name, value, unit, reference range
            parsed_labs = []
            # Try common lab report patterns
            patterns = [
                # Pattern: Name  Value  Unit  Low-High
                r'([A-Za-z\s/\-]+?)\s+([\d.]+)\s+([A-Za-z/%\u00b3\u00b5\u2076\s]+?)\s+([\d.]+)\s*[-\u2013]\s*([\d.]+)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, raw_text)
                for match in matches:
                    try:
                        analyte = match[0].strip()
                        value = float(match[1])
                        unit = match[2].strip()
                        ref_low = float(match[3])
                        ref_high = float(match[4])
                        parsed_labs.append({
                            "analyte": analyte,
                            "value": value,
                            "unit": unit,
                            "ref_low": ref_low,
                            "ref_high": ref_high,
                        })
                    except (ValueError, IndexError):
                        continue

            if parsed_labs:
                lab_data = parsed_labs
            else:
                lab_data = DEMO_LAB_REPORT
                used_demo = True
                st.warning("Could not parse lab values from PDF. Showing demo data for illustration.")

        except ImportError:
            st.warning("pdfplumber is not installed. Install with: pip install pdfplumber. Showing demo data.")
            lab_data = DEMO_LAB_REPORT
            used_demo = True
            raw_text = "(PDF parsing unavailable — pdfplumber not installed)"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            lab_data = DEMO_LAB_REPORT
            used_demo = True
    else:
        lab_data = DEMO_LAB_REPORT
        used_demo = True

    with tab_parsed:
        if lab_data:
            # Classify each value
            results = []
            for item in lab_data:
                status, css_class, color = classify_value(
                    item["value"], item["ref_low"], item["ref_high"]
                )
                results.append({**item, "status": status, "css_class": css_class, "color": color})

            # Summary metrics
            total_tests = len(results)
            normal_count = sum(1 for r in results if r["status"] == "Normal")
            abnormal_count = total_tests - normal_count

            sc1, sc2, sc3 = st.columns(3, gap="medium")
            with sc1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Total Tests</p>
                    <p class="metric-value">{total_tests}</p>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="metric-card risk-low">
                    <p class="metric-label">Normal</p>
                    <p class="metric-value">{normal_count}</p>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                abn_class = "risk-high" if abnormal_count > 0 else "risk-low"
                st.markdown(f"""
                <div class="metric-card {abn_class}">
                    <p class="metric-label">Abnormal</p>
                    <p class="metric-value">{abnormal_count}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if used_demo:
                st.markdown("""
                <div class="info-card">
                    <h4>Demo Lab Report</h4>
                    <p>Displaying sample lab data for demonstration. Upload a PDF to analyze your own report.</p>
                </div>
                """, unsafe_allow_html=True)

            # Styled HTML table
            table_rows = ""
            for r in results:
                table_rows += f"""
                <tr>
                    <td style="font-weight:600;">{r['analyte']}</td>
                    <td>{r['value']}</td>
                    <td>{r['unit']}</td>
                    <td>{r['ref_low']} - {r['ref_high']}</td>
                    <td><span class="{r['css_class']}">{r['status']}</span></td>
                </tr>"""

            st.markdown(f"""
            <table class="lab-table">
                <thead>
                    <tr>
                        <th>Analyte</th>
                        <th>Value</th>
                        <th>Unit</th>
                        <th>Reference Range</th>
                        <th>Flag</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Abnormal values bar chart
            abnormal_results = [r for r in results if r["status"] != "Normal"]
            if abnormal_results:
                st.markdown("**Abnormal Values — Deviation from Reference Range**")
                abn_names = []
                abn_deviations = []
                abn_colors = []
                for r in abnormal_results:
                    abn_names.append(r["analyte"])
                    if r["value"] > r["ref_high"]:
                        dev = r["value"] - r["ref_high"]
                    else:
                        dev = r["ref_low"] - r["value"]
                    abn_deviations.append(round(dev, 2))
                    abn_colors.append(r["color"])

                fig_abn = go.Figure(data=[go.Bar(
                    x=abn_names, y=abn_deviations,
                    marker_color=abn_colors,
                    text=[f"+{d}" if d > 0 else str(d) for d in abn_deviations],
                    textposition="outside",
                    textfont=dict(family="Inter", size=12),
                )])
                fig_abn.update_layout(
                    height=320,
                    template="plotly_white",
                    font=dict(family="Inter"),
                    yaxis_title="Deviation from Reference",
                    xaxis_title="",
                    margin=dict(t=40, b=20, l=20, r=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_abn, use_container_width=True)

        st.markdown("""
        <div class="disclaimer">
            <strong>Disclaimer:</strong> Automated PDF parsing may be inaccurate. Values extracted from uploaded
            reports should be verified against the original document. This tool is for educational and screening
            purposes only and does not replace professional laboratory interpretation.
        </div>
        """, unsafe_allow_html=True)

    with tab_raw:
        if raw_text:
            st.text_area("Raw Extracted Text", raw_text, height=400)
        else:
            st.info("Upload a PDF to see the raw extracted text. Demo mode does not have raw text.")

        st.markdown("""
        <div class="disclaimer">
            <strong>Disclaimer:</strong> Automated PDF parsing may be inaccurate. Always verify extracted values
            against the original document.
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
