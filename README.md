# Healthcare AI Prediction Portal
## Abdullah Abdul Sami — Northwestern University MSDS (AI Track)

A multi-model healthcare prediction platform that lets users upload medical data (ECG signals, chest X-rays, vital signs) and receive AI-powered diagnostic predictions with interactive dashboards.

---

## Project Structure

```
healthcare-ai-portal/
├── data/
│   ├── ecg/              # PTB-XL ECG dataset files
│   ├── xray/             # Chest X-ray dataset files
│   └── vitals/           # Vital signs / tabular health data
├── models/               # Saved trained models (.h5, .pkl, .joblib)
├── notebooks/
│   ├── 01_ecg_data_prep.ipynb
│   ├── 02_ecg_model_training.ipynb
│   ├── 03_xray_model_training.ipynb
│   └── 04_vitals_model_training.ipynb
├── app/
│   ├── streamlit_app.py  # Main web application
│   ├── static/           # CSS, images
│   └── templates/        # HTML templates (if using Flask)
├── scripts/
│   ├── download_data.sh  # Dataset download script
│   └── train_all.py      # Train all models sequentially
├── utils/
│   ├── ecg_utils.py      # ECG preprocessing functions
│   ├── xray_utils.py     # X-ray preprocessing functions
│   └── model_utils.py    # Model loading and prediction helpers
├── requirements.txt
└── README.md
```

---

## Datasets Used

### 1. PTB-XL (ECG) — PRIMARY
- **Source**: https://physionet.org/content/ptb-xl/1.0.3/
- **Size**: 21,837 clinical 12-lead ECGs (10 seconds each)
- **Labels**: 71 diagnostic statements (normal, MI, STEMI, AFib, etc.)
- **Format**: WFDB format (.dat + .hea files)
- **License**: Open Data Commons Attribution License
- **No credentialing required** — direct download

### 2. NIH Chest X-ray14 (X-Ray)
- **Source**: https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Size**: 112,120 frontal-view X-ray images
- **Labels**: 14 disease labels + "No Finding"
- **Format**: PNG images
- **License**: CC0 Public Domain

### 3. Heart Disease UCI (Vitals/Tabular)
- **Source**: https://archive.ics.uci.edu/dataset/45/heart+disease
- **Size**: 920 patient records
- **Features**: Age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Format**: CSV
- **Good for**: Quick risk prediction dashboard

---

## Setup Instructions

### Step 1: Clone and setup environment
```bash
cd healthcare-ai-portal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download datasets
```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

### Step 3: Train models (or use notebooks)
```bash
# Option A: Run notebooks in Jupyter/Colab
jupyter notebook notebooks/

# Option B: Run training script
python scripts/train_all.py
```

### Step 4: Launch the web app
```bash
streamlit run app/streamlit_app.py
```

---

## For Google Colab Workflow
1. Upload the `notebooks/` folder to Google Drive
2. Open each notebook in Colab
3. Train models — they auto-save to Google Drive `/models/`
4. Download saved models to local `models/` folder
5. Run the Streamlit app locally pointing to those models

---

## Author
Abdullah Abdul Sami
MS in Data Science (AI Specialization) — Northwestern University
Research: Edge-AI for Cardiovascular Monitoring
