#!/bin/bash
# ============================================================
# Healthcare AI Portal — Dataset Download Script
# Run this from the project root: ./scripts/download_data.sh
# ============================================================

set -e
echo "============================================"
echo "Healthcare AI Portal — Dataset Downloader"
echo "============================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# -------------------------------------------------------
# 1. PTB-XL ECG Dataset (21,837 clinical 12-lead ECGs)
# -------------------------------------------------------
echo "[1/3] Downloading PTB-XL ECG Dataset..."
ECG_DIR="$PROJECT_ROOT/data/ecg"
mkdir -p "$ECG_DIR"

if [ -f "$ECG_DIR/ptbxl_database.csv" ]; then
    echo "  PTB-XL already downloaded. Skipping."
else
    echo "  Downloading from PhysioNet (approx 1.8 GB)..."
    echo "  This may take a few minutes depending on your connection."
    wget -q --show-progress -O "$ECG_DIR/ptb-xl-1.0.3.zip" \
        "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    echo "  Extracting..."
    cd "$ECG_DIR"
    unzip -q ptb-xl-1.0.3.zip
    # Move files from nested folder to ecg/ root
    mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/* .
    rmdir ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
    rm ptb-xl-1.0.3.zip
    echo "  PTB-XL downloaded and extracted."
fi
echo ""

# -------------------------------------------------------
# 2. Heart Disease UCI Dataset (tabular vital signs)
# -------------------------------------------------------
echo "[2/3] Downloading Heart Disease UCI Dataset..."
VITALS_DIR="$PROJECT_ROOT/data/vitals"
mkdir -p "$VITALS_DIR"

if [ -f "$VITALS_DIR/heart.csv" ]; then
    echo "  Heart Disease dataset already downloaded. Skipping."
else
    echo "  Downloading from UCI..."
    # Use the processed Cleveland dataset from UCI via a known mirror
    wget -q --show-progress -O "$VITALS_DIR/heart.csv" \
        "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
    echo "  Heart Disease UCI dataset downloaded."
fi
echo ""

# -------------------------------------------------------
# 3. Chest X-ray (Pneumonia) Dataset
# -------------------------------------------------------
echo "[3/3] Chest X-ray Dataset..."
XRAY_DIR="$PROJECT_ROOT/data/xray"
mkdir -p "$XRAY_DIR"

echo "  The NIH Chest X-ray14 dataset is 42 GB."
echo "  For a faster start, we recommend the Kaggle Chest X-Ray (Pneumonia) dataset:"
echo ""
echo "  OPTION A (Recommended for quick start):"
echo "    1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
echo "    2. Download and extract to: $XRAY_DIR/"
echo "    3. You should have: $XRAY_DIR/train/, $XRAY_DIR/val/, $XRAY_DIR/test/"
echo ""
echo "  OPTION B (Full NIH dataset — 42 GB):"
echo "    1. Go to: https://nihcc.app.box.com/v/ChestXray-NIHCC"
echo "    2. Download image batches to: $XRAY_DIR/"
echo ""

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo "============================================"
echo "Download Summary"
echo "============================================"
echo ""

check_dir() {
    if [ -d "$1" ] && [ "$(ls -A $1 2>/dev/null)" ]; then
        echo "  [OK]  $2"
    else
        echo "  [--]  $2 (empty or missing)"
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK]  $2"
    else
        echo "  [--]  $2 (not found)"
    fi
}

check_file "$ECG_DIR/ptbxl_database.csv" "PTB-XL ECG Dataset"
check_file "$VITALS_DIR/heart.csv" "Heart Disease UCI Dataset"
check_dir "$XRAY_DIR/train" "Chest X-ray Dataset"

echo ""
echo "Next steps:"
echo "  1. Open notebooks/ in Jupyter or Google Colab"
echo "  2. Run 01_ecg_data_prep.ipynb first"
echo "  3. Then train models with 02, 03, 04 notebooks"
echo "  4. Launch app: streamlit run app/streamlit_app.py"
echo ""
echo "Done!"
