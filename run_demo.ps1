# =============================================================================
# HemoVision — Automated Setup and Run Script (PowerShell)
# =============================================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  HemoVision - Contactless Vital Signs (rPPG) System       " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python installation
try {
    $pythonVer = python --version 2>&1
    Write-Host "[+] Detected Python: $pythonVer" -ForegroundColor Green
} catch {
    Write-Host "[!] Error: Python is not detected in your PATH." -ForegroundColor Red
    Write-Host "    Please install Python (Python 3.10 is strongly recommended) from python.org"
    Write-Host "    Make sure to check 'Add Python to PATH' during installation."
    Exit 1
}

# 2. Setup Virtual Environment if not present
$venvDir = "venv"
if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
    Write-Host "[*] Creating fresh virtual environment in .\$venvDir..." -ForegroundColor Yellow
    python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }
}

# 3. Activate Virtual Environment
Write-Host "[*] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = "$venvDir\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "[!] Could not find $activateScript. Attempting direct execution..." -ForegroundColor Yellow
}

# 4. Install Dependencies
$pythonExe = "$venvDir\Scripts\python.exe"
Write-Host "[*] Ensuring dependencies are installed from requirements.txt..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Dependency installation encountered an issue." -ForegroundColor Red
    Write-Host "    Attempting to run anyway..." -ForegroundColor Yellow
} else {
    Write-Host "[+] All dependencies verified successfully!" -ForegroundColor Green
}

# 5. Launch Menu
Write-Host ""
Write-Host "Select an option to launch:" -ForegroundColor Cyan
Write-Host "  [1] Streamlit Web Dashboard (Interactive Presentation UI)" -ForegroundColor White
Write-Host "  [2] Live Webcam Demo (Real-Time Face Mesh & rPPG Inference)" -ForegroundColor White
Write-Host "  [3] Offline Test Clip Comparison (Benchmark vs Ground Truth)" -ForegroundColor White
Write-Host "  [4] Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice [1-4] (Default: 1)"

if ([string]::IsNullOrWhiteSpace($choice)) {
    $choice = "1"
}

switch ($choice) {
    "1" {
        Write-Host "[*] Starting Streamlit Web App..." -ForegroundColor Green
        & "$venvDir\Scripts\streamlit.exe" run app.py
    }
    "2" {
        Write-Host "[*] Starting Live Webcam Inference..." -ForegroundColor Green
        & $pythonExe hemovision_demo.py
    }
    "3" {
        Write-Host "[*] Running Test Set Comparison..." -ForegroundColor Green
        & $pythonExe hemovision_demo.py --test
    }
    default {
        Write-Host "[*] Exiting. Have a great presentation!" -ForegroundColor Cyan
    }
}
