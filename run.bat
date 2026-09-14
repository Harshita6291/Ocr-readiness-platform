@echo off
setlocal enabledelayedexpansion
title OCR Readiness Platform - Auto Launcher

echo ========================================================
echo   OCR Readiness Evaluation Platform - Setup and Launcher
echo ========================================================
echo.

:: 1. Navigate to script directory
cd /d "%~dp0"

:: 2. Check if virtual environment already exists
if exist ".venv\Scripts\activate.bat" (
    echo [✓] Using project virtual environment...
    call .venv\Scripts\activate.bat
    goto :post_venv
)

:: Find Python executable to create .venv
set PYTHON_CMD=
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
) else (
    where py >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        set PYTHON_CMD=py
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.9+ from https://www.python.org/ and check "Add Python to PATH".
    pause
    exit /b 1
)

echo [✓] Found Python: %PYTHON_CMD%

:: 3. Create virtual environment if it does not exist
echo [i] Creating virtual environment (.venv)...
%PYTHON_CMD% -m venv .venv
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [✓] Virtual environment created successfully.

:: 4. Activate virtual environment
call .venv\Scripts\activate.bat

:post_venv

:: 5. Install / Update requirements
echo [i] Checking and installing dependencies from requirements.txt...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

:: 6. Check Tesseract OCR
echo [i] Checking Tesseract OCR...
if exist "%CD%\tesseract\tesseract.exe" (
    set "PATH=%CD%\tesseract;!PATH!"
    set "TESSERACT_CMD=%CD%\tesseract\tesseract.exe"
    echo [✓] Configured local Tesseract: %CD%\tesseract\tesseract.exe
)
if exist "%CD%\tessdata" (
    set "TESSDATA_PREFIX=%CD%\tessdata"
    echo [✓] Configured local tessdata: %CD%\tessdata
)

:: 7. Launch Streamlit Application
echo.
echo ========================================================
echo   Starting Streamlit App at http://localhost:8501
echo ========================================================
echo.

python -m streamlit run app.py

pause
