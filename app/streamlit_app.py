"""
Healthcare AI Prediction Portal
================================
Multi-section dashboard for medical data analysis and prediction.

Sections:
1. Heart / ECG Analysis — Upload ECG, get arrhythmia classification
2. Chest X-Ray Analysis — Upload X-ray, get pneumonia/disease prediction
3. Health Risk Assessment — Input vitals, get heart disease risk score

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

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

    /* ── Hide Streamlit defaults ────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar Navigation
# ============================================================
with st.sidebar:
    st.markdown("### Healthcare AI")
    st.markdown("**Prediction Portal**")
    st.markdown("---")

    section = st.radio(
        "Navigation",
        ["Home", "Heart / ECG", "Chest X-Ray", "Health Risk Assessment"],
        index=0,
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
            <span class="stat-pill"><strong>3</strong>&nbsp; AI Models</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon icon-ecg">❤️</div>
            <h3>Heart / ECG Analysis</h3>
            <p>
                Upload 12-lead ECG recordings for real-time arrhythmia classification.
                Get heart rate, HRV metrics, and diagnostic probabilities across 5 cardiac conditions.
            </p>
            <span class="feature-tag">1D CNN Model</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon icon-xray">🫁</div>
            <h3>Chest X-Ray Analysis</h3>
            <p>
                Upload frontal chest X-ray images for pneumonia detection and
                multi-label disease classification with visual attention maps.
            </p>
            <span class="feature-tag">Transfer Learning</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon icon-risk">📊</div>
            <h3>Health Risk Assessment</h3>
            <p>
                Enter patient vitals and clinical data to generate a heart disease
                risk score with interactive gauge, radar chart, and factor analysis.
            </p>
            <span class="feature-tag">Random Forest</span>
        </div>
        """, unsafe_allow_html=True)

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
