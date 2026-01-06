#!/bin/bash

# KitchenMind API Startup Script for Linux/Mac

echo ""
echo "===================================================="
echo "   KitchenMind API - FastAPI + PostgreSQL"
echo "===================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8 or higher."
    echo "Download from: https://www.python.org/downloads/"
    exit 1
fi

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Setup database
echo "Setting up PostgreSQL database..."
python setup_db.py
echo ""

# Run FastAPI server
echo "Starting FastAPI server..."
echo ""
echo "Server running at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
