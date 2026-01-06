@echo off
REM KitchenMind API Startup Script for Windows

echo.
echo ====================================================
echo     KitchenMind API - FastAPI + PostgreSQL
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if venv exists, if not create it
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Setup database
echo Setting up PostgreSQL database...
python setup_db.py
echo.

REM Run FastAPI server
echo Starting FastAPI server...
echo.
echo Server running at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

pause
