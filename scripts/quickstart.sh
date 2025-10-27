#!/bin/bash
# Quick start script for ATHENA MLOps Platform (Unix/macOS/Linux)

echo "=========================================="
echo "ATHENA MLOps Platform - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize storage
echo ""
echo "Initializing storage..."
python -m athena --check-storage

# Run setup check
echo ""
echo "Running setup verification..."
python scripts/setup_check.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Install Ollama: https://ollama.ai"
echo "3. Pull Llama model: ollama pull llama3.1:8b"
echo "4. Run sample training: python scripts/train_mnist.py"
echo "=========================================="
