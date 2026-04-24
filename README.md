# Healthcare AI Prediction Portal

A Streamlit platform bundling nine clinical decision-support modules — ECG analysis, chest X-ray classification, vitals-based risk prediction, and an LLM-backed clinical chat assistant — behind a single web interface.

**Live demo:** https://healthcare-ai-app-5n6p29wbkektzs8dkl86qr.streamlit.app

---

## Overview

The portal wraps a set of independent clinical models and clinical rule-based tools in a shared Streamlit shell. Each module exposes a clinician-facing interface: file upload for signals and imaging, guided questionnaires for risk scores, and a conversational layer for portal navigation and result interpretation. Modules share session state, theming, and a common evaluation harness; the LLM assistant is aware of the portal's structure so it can route users to the right tool.

The application is deployed on Streamlit Cloud and backed by a small collection of trained artifacts (`.joblib`, `.h5`) loaded at runtime.

---

## Features

Nine clinical decision-support modules plus a HIPAA/compliance reference page:

1. **Heart / ECG Analysis** — 1D-CNN for five-class ECG classification (NORM, MI, STTC, HYP, CD) over a 12-lead, 10-second window. Note: the included training pipeline uses synthetic signals (`scripts/generate_ecg_data.py`) as a demonstration dataset; the model has not been evaluated on real clinical ECGs.
2. **Chest X-Ray Analysis** — MobileNetV2 transfer-learning scaffold (pneumonia vs normal). The training pipeline (`scripts/train_all.py`) targets the Kaggle Chest X-Ray Pneumonia dataset. No trained artifact is checked into this repository; the app detects the absence and falls back gracefully.
3. **Health Risk Assessment** — Soft-voting ensemble (XGBoost + LightGBM + Gradient Boosting) over engineered features from the combined UCI Heart Disease and Framingham cohorts (≈5,160 records), with SMOTE oversampling. Held-out test metrics (15% stratified split): accuracy 0.846, F1 0.544, AUC-ROC 0.786.
4. **CBC Analysis** — Rule-based interpretation of complete blood count values with WBC differentials and clinical flagging.
5. **Diabetes Screening** — FINDRISC questionnaire combined with HbA1c and fasting-glucose threshold logic.
6. **Lipid Panel / Cardiovascular Risk** — ATP III lipid classification plus 10-year ASCVD risk via the Pooled Cohort Equations.
7. **Kidney Function** — Race-free eGFR using the CKD-EPI 2021 equation with KDIGO staging and albuminuria risk classification.
8. **Lab Report Upload** — PDF parser for uploaded lab reports, with automated value extraction and abnormal-flag highlighting.
9. **AI Assistant** — Portal-aware chat interface backed by Anthropic's Messages API. Scoped to portal navigation and result interpretation; does not provide medical advice.

A tenth page, **Privacy & Compliance**, documents HIPAA-relevant handling, data minimization, and the portal's scope limits. It is a reference page, not a predictive module.

---

## Tech Stack

- **Application:** Python 3.11, Streamlit
- **Modeling:** TensorFlow / Keras (ECG, X-ray), scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE)
- **Data handling:** pandas, NumPy, WFDB
- **Visualisation:** Plotly, Matplotlib
- **LLM integration:** `anthropic` Python SDK
- **Artifact persistence:** joblib, `.h5` / Keras serialisation

Pinned versions are listed in `requirements.txt`.

---

## Local Setup

```bash
git clone https://github.com/abdullahabdulsami2026-coder/healthcare-ai-portal.git
cd healthcare-ai-portal

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

The app will start on `http://localhost:8501`.

### Optional: retrain the models

Trained artifacts live under `models/` and are loaded at runtime. To reproduce them from source:

```bash
# Synthetic ECG dataset (used for the demo ECG model)
python scripts/generate_ecg_data.py

# Heart risk model (UCI + Framingham, held-out test + full-data fit)
python scripts/retrain_heart_model.py
```

### Optional: enable the AI Assistant

Set an Anthropic API key before launching Streamlit:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or add it to `.streamlit/secrets.toml` (gitignored).

---

## Repository Structure

```
healthcare-ai-portal/
├── app/
│   └── streamlit_app.py          # Single-file Streamlit application
├── data/
│   ├── ecg/                      # Synthetic ECG arrays (generated, gitignored)
│   ├── vitals/                   # Tabular clinical datasets (UCI, Framingham)
│   └── xray/                     # Chest X-ray samples
├── models/                       # Trained model artifacts (.joblib, .h5)
├── notebooks/                    # Exploratory notebooks
├── scripts/                      # Data generation + training pipelines
├── utils/                        # Clinical interpretation + design helpers
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Author

**Abdullah Abdul Sami** — MS Data Science, Northwestern University

---

## License

This project is released under the MIT License. See `LICENSE` for details.

The bundled clinical rule implementations (CKD-EPI 2021, ATP III, Pooled Cohort Equations, FINDRISC) reference published clinical guidelines. The portal is a technical demonstration and is not a substitute for medical advice, diagnosis, or treatment.
