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
# Custom CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    .main { background-color: #f8fafb; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid #1a5276;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin: 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .risk-high { border-left-color: #e74c3c !important; }
    .risk-medium { border-left-color: #f39c12 !important; }
    .risk-low { border-left-color: #27ae60 !important; }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a5276;
        border-bottom: 2px solid #1a5276;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    .result-box {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin: 16px 0;
    }

    .result-box h2 { color: white; margin: 0; font-size: 1.8rem; }
    .result-box p { color: rgba(255,255,255,0.85); margin: 4px 0 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar Navigation
# ============================================================
with st.sidebar:
    st.markdown("# 🏥 Healthcare AI")
    st.markdown("**Prediction Portal**")
    st.markdown("---")

    section = st.radio(
        "Select Analysis",
        ["🏠 Home", "❤️ Heart / ECG", "🫁 Chest X-Ray", "📊 Health Risk Assessment"],
        index=0,
    )

    st.markdown("---")
    st.markdown("##### About")
    st.markdown(
        "Built by **Abdullah Abdul Sami**\n\n"
        "MS Data Science (AI)\n\n"
        "Northwestern University"
    )
    st.markdown("---")
    st.caption("Models trained on public datasets. Not for clinical use.")


# ============================================================
# HOME SECTION
# ============================================================
if section == "🏠 Home":
    st.markdown("# 🏥 Healthcare AI Prediction Portal")
    st.markdown(
        "Upload medical data and get AI-powered predictions with interactive dashboards. "
        "Select a section from the sidebar to begin."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Heart / ECG</p>
            <p class="metric-value">❤️</p>
            <p style="color:#666; font-size:0.9rem;">
                Upload 12-lead ECG signals. Get arrhythmia classification,
                heart rate analysis, and HRV metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Chest X-Ray</p>
            <p class="metric-value">🫁</p>
            <p style="color:#666; font-size:0.9rem;">
                Upload chest X-ray images. Get pneumonia detection
                and multi-label disease classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Health Risk</p>
            <p class="metric-value">📊</p>
            <p style="color:#666; font-size:0.9rem;">
                Input vital signs and patient data. Get heart disease
                risk prediction with interactive charts.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Datasets Used")
    st.markdown(
        "- **PTB-XL**: 21,837 clinical 12-lead ECGs from PhysioNet\n"
        "- **NIH Chest X-ray14**: 112,120 frontal-view chest X-rays\n"
        "- **UCI Heart Disease**: 920 patient records with diagnostic labels"
    )


# ============================================================
# HEART / ECG SECTION
# ============================================================
elif section == "❤️ Heart / ECG":
    st.markdown('<p class="section-header">❤️ Heart / ECG Analysis</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload ECG", "📈 Demo Analysis"])

    # --- Upload Tab ---
    with tab1:
        st.markdown("Upload a 12-lead ECG recording (.csv, .dat, or .hea format)")
        uploaded_file = st.file_uploader(
            "Choose ECG file", type=["csv", "dat", "hea", "npy"],
            help="Supported formats: CSV (columns = leads), NumPy arrays, or WFDB format"
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
                    st.success(f"Loaded ECG signal: {ecg_data.shape[0]} samples × {ecg_data.shape[1] if ecg_data.ndim > 1 else 1} leads")

                    # Plot the signal
                    fig = go.Figure()
                    if ecg_data.ndim > 1:
                        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
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
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Try to run prediction
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

                        # Probability chart
                        probs_df = pd.DataFrame(
                            list(result["all_probabilities"].items()),
                            columns=["Condition", "Probability (%)"]
                        )
                        fig_probs = px.bar(
                            probs_df, x="Probability (%)", y="Condition",
                            orientation="h", color="Probability (%)",
                            color_continuous_scale="Blues"
                        )
                        fig_probs.update_layout(height=300, template="plotly_white")
                        st.plotly_chart(fig_probs, use_container_width=True)

                    except FileNotFoundError:
                        st.info("⚠️ ECG model not trained yet. Train the model using `notebooks/02_ecg_model_training.ipynb` first.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

            except Exception as e:
                st.error(f"Error loading file: {e}")

    # --- Demo Tab ---
    with tab2:
        st.markdown("### Demo: Simulated ECG Analysis")
        st.markdown("See what the dashboard looks like with sample data.")

        # Generate synthetic ECG
        np.random.seed(42)
        fs = 500
        t = np.linspace(0, 10, fs * 10)

        # Simulate a rough ECG-like signal
        ecg_sim = (
            0.6 * np.sin(2 * np.pi * 1.2 * t) +
            0.3 * np.sin(2 * np.pi * 2.4 * t) +
            np.where((t % (1/1.2)) < 0.05, 1.5, 0) +
            0.05 * np.random.randn(len(t))
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><p class="metric-label">Heart Rate</p><p class="metric-value">72 BPM</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><p class="metric-label">SDNN</p><p class="metric-value">45.2 ms</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><p class="metric-label">RMSSD</p><p class="metric-value">38.7 ms</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card risk-low"><p class="metric-label">Classification</p><p class="metric-value" style="font-size:1.4rem;">Normal</p></div>', unsafe_allow_html=True)

        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(
            x=t[:2000], y=ecg_sim[:2000],
            mode="lines", name="Lead II",
            line=dict(color="#1a5276", width=1.5)
        ))
        fig_demo.update_layout(
            title="12-Lead ECG — Lead II",
            xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
            height=400, template="plotly_white"
        )
        st.plotly_chart(fig_demo, use_container_width=True)

        # Demo probability chart
        demo_probs = pd.DataFrame({
            "Condition": ["Normal ECG", "ST/T Change", "Conduction Dist.", "Hypertrophy", "MI"],
            "Probability (%)": [89.2, 5.1, 3.2, 1.8, 0.7]
        })
        fig_p = px.bar(
            demo_probs, x="Probability (%)", y="Condition",
            orientation="h", color="Probability (%)",
            color_continuous_scale="Blues"
        )
        fig_p.update_layout(height=280, template="plotly_white")
        st.plotly_chart(fig_p, use_container_width=True)


# ============================================================
# CHEST X-RAY SECTION
# ============================================================
elif section == "🫁 Chest X-Ray":
    st.markdown('<p class="section-header">🫁 Chest X-Ray Analysis</p>', unsafe_allow_html=True)
    st.markdown("Upload a frontal chest X-ray image for AI-powered analysis.")

    uploaded_xray = st.file_uploader(
        "Choose X-ray image", type=["png", "jpg", "jpeg", "dcm"],
        help="Supported: PNG, JPEG. Frontal PA/AP view recommended."
    )

    if uploaded_xray is not None:
        from PIL import Image

        img = Image.open(uploaded_xray).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption="Uploaded X-ray", use_container_width=True)

        with col2:
            try:
                from utils.model_utils import load_model, predict_with_confidence
                from utils.xray_utils import (
                    load_and_preprocess_xray, prepare_xray_for_model,
                    PNEUMONIA_CLASSES, XRAY_CLASSES
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
                    <p style="color:#666;">Confidence: {result['confidence']}%</p>
                </div>
                """, unsafe_allow_html=True)

                probs_df = pd.DataFrame(
                    list(result["all_probabilities"].items()),
                    columns=["Class", "Probability (%)"]
                )
                fig = px.pie(probs_df, values="Probability (%)", names="Class",
                             color_discrete_sequence=["#27ae60", "#e74c3c"])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            except FileNotFoundError:
                st.info("⚠️ X-ray model not trained yet. Train using `notebooks/03_xray_model_training.ipynb`.")

                # Show demo results instead
                st.markdown("**Demo prediction (model not loaded):**")
                demo_data = pd.DataFrame({
                    "Condition": ["Normal", "Pneumonia"],
                    "Probability (%)": [82.3, 17.7]
                })
                fig = px.pie(demo_data, values="Probability (%)", names="Condition",
                             color_discrete_sequence=["#27ae60", "#e74c3c"])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.markdown("*Upload a chest X-ray to get started.*")


# ============================================================
# HEALTH RISK ASSESSMENT SECTION
# ============================================================
elif section == "📊 Health Risk Assessment":
    st.markdown('<p class="section-header">📊 Health Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown("Enter patient vitals to get a heart disease risk prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [
            "Typical Angina", "Atypical Angina",
            "Non-Anginal Pain", "Asymptomatic"
        ])
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 220, 130)

    with col2:
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", [
            "Normal", "ST-T Abnormality", "LV Hypertrophy"
        ])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

    with col3:
        exang = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
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

        # Try loading trained model
        try:
            from utils.model_utils import load_model, predict_with_confidence
            model = load_model("heart_risk", MODELS_DIR)
            result = predict_with_confidence(model, features, class_names=["Low Risk", "High Risk"])
            risk_score = result["all_probabilities"].get("High Risk", 0)
            predicted = result["predicted_label"]
        except FileNotFoundError:
            # Fallback: simple rule-based risk estimate for demo
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

            st.info("⚠️ Using rule-based estimate. Train the model with `notebooks/04_vitals_model_training.ipynb` for ML predictions.")

        # Display results
        st.markdown("---")
        st.markdown("### Results")

        risk_class = "risk-high" if predicted == "High Risk" else "risk-low"

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <p class="metric-label">Risk Level</p>
                <p class="metric-value">{predicted}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Risk Score</p>
                <p class="metric-value">{risk_score:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Heart Rate</p>
                <p class="metric-value">{thalach} BPM</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Heart Disease Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1a5276"},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": risk_score,
                },
            },
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Patient vitals radar chart
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
            fillcolor="rgba(26,82,118,0.2)",
            line=dict(color="#1a5276"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Patient Vitals Overview",
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Healthcare AI Prediction Portal | Abdullah Abdul Sami | Northwestern University | "
    "For research and educational purposes only. Not for clinical diagnosis."
    "</p>",
    unsafe_allow_html=True,
)
