@echo off
REM Quick start script for ATHENA MLOps Platform (Windows)

echo ==========================================
echo ATHENA MLOps Platform - Quick Start
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Initialize storage
echo Initializing storage...
python -m athena --check-storage
echo.

REM Run setup check
echo Running setup verification...
python scripts\setup_check.py
echo.

echo ==========================================
echo Setup complete!
echo.
echo Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Install Ollama: https://ollama.ai
echo 3. Pull Llama model: ollama pull llama3.1:8b
echo 4. Run sample training: python scripts\train_mnist.py
echo ==========================================
pause
