@echo off
title HemoVision - Contactless Vital Signs Setup & Runner
color 0B

echo ============================================================
echo   HemoVision - Contactless Vital Signs (rPPG) System       
echo ============================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python was not found on your system PATH!
    echo Please install Python 3.10 from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

if not exist venv\Scripts\python.exe (
    echo [*] Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [*] Checking dependencies from requirements.txt...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ============================================================
echo   Select launch mode:
echo     [1] Streamlit Web Dashboard (Recommended for presentation)
echo     [2] Live Webcam Demo (Real-time rPPG)
echo     [3] Offline Test Clip Comparison
echo ============================================================
echo.

set /p choice="Enter choice [1-3] (Default: 1): "
if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    echo [*] Starting Streamlit App...
    venv\Scripts\streamlit.exe run app.py
) else if "%choice%"=="2" (
    echo [*] Starting Webcam Demo...
    venv\Scripts\python.exe hemovision_demo.py
) else if "%choice%"=="3" (
    echo [*] Starting Test Comparison...
    venv\Scripts\python.exe hemovision_demo.py --test
) else (
    echo Invalid choice. Starting Streamlit by default...
    venv\Scripts\streamlit.exe run app.py
)

pause
