#!/bin/bash
# =============================================================================
# HemoVision — Automated Setup and Run Script (macOS / Linux)
# =============================================================================

set -e

echo "============================================================"
echo "  HemoVision - Contactless Vital Signs (rPPG) System        "
echo "  (macOS / Linux Edition)                                   "
echo "============================================================"
echo ""

# 1. Detect Python 3
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[-] Error: Python is not installed or not in PATH."
    echo "    Please install Python 3.10+ (via python.org or 'brew install python@3.10')."
    exit 1
fi

echo "[+] Using: $($PYTHON_CMD --version)"

# 2. Create virtual environment if not present
VENV_DIR="venv"
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[*] Creating fresh virtual environment in ./$VENV_DIR..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

# 3. Activate virtual environment
echo "[*] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 4. Install dependencies
echo "[*] Verifying dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[+] Setup completed successfully!"
echo ""
echo "Select an option to launch:"
echo "  [1] Streamlit Web Dashboard (Recommended for Presentation)"
echo "  [2] Live Webcam Demo (Real-Time Face Mesh & rPPG)"
echo "  [3] Offline Test Clip Benchmark"
echo "  [4] Exit"
echo ""

read -p "Enter choice [1-4] (Default: 1): " choice
choice=${choice:-1}

case "$choice" in
    1)
        echo "[*] Launching Streamlit Web App..."
        streamlit run app.py
        ;;
    2)
        echo "[*] Launching Live Webcam Demo..."
        python hemovision_demo.py
        ;;
    3)
        echo "[*] Launching Test Clip Benchmark..."
        python hemovision_demo.py --test
        ;;
    *)
        echo "[*] Exiting. Good luck with the presentation!"
        ;;
esac
