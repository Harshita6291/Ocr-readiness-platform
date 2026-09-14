#!/usr/bin/env bash
# OCR Readiness Platform - Auto Launcher for Linux & macOS

set -e

echo "========================================================"
echo "  OCR Readiness Evaluation Platform - Setup & Launcher"
echo "========================================================"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[i] Creating virtual environment (.venv)..."
    python3 -m venv .venv
    echo "[✓] Virtual environment created."
fi

source .venv/bin/activate

echo "[i] Installing dependencies..."
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if [ -d "$SCRIPT_DIR/tessdata" ]; then
    export TESSDATA_PREFIX="$SCRIPT_DIR/tessdata"
    echo "[✓] TESSDATA_PREFIX set to $TESSDATA_PREFIX"
fi

echo ""
echo "========================================================"
echo "  Starting Streamlit App at http://localhost:8501"
echo "========================================================"
echo ""

python -m streamlit run app.py
