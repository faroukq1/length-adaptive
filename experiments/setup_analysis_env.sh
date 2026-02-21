#!/bin/bash

# Check if virtual environment exists, create if not
# Usage: source experiments/setup_analysis_env.sh

VENV_DIR="venv"

echo "===================================================================="
echo "ANALYSIS ENVIRONMENT SETUP"
echo "===================================================================="

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Virtual environment not found. Creating..."
    echo ""
    
    python3 -m venv $VENV_DIR
    
    if [ $? -eq 0 ]; then
        echo "✓ Virtual environment created successfully"
    else
        echo "❌ Failed to create virtual environment"
        echo "   Make sure python3-venv is installed:"
        echo "   sudo apt install python3-venv"
        exit 1
    fi
fi

# Activate venv
echo ""
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Check for required packages
echo ""
echo "Checking required packages..."

PACKAGES_MISSING=0

python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ⚠️  torch not installed"
    PACKAGES_MISSING=1
fi

python -c "import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ⚠️  matplotlib not installed"
    PACKAGES_MISSING=1
fi

python -c "import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ⚠️  tqdm not installed"
    PACKAGES_MISSING=1
fi

if [ $PACKAGES_MISSING -eq 1 ]; then
    echo ""
    echo "Installing missing packages..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -q matplotlib seaborn tqdm
    
    echo "✓ Packages installed"
fi

echo ""
echo "===================================================================="
echo "✅ ENVIRONMENT READY"
echo "===================================================================="
echo ""
echo "You can now run analysis scripts:"
echo "  python experiments/quick_compare.py"
echo "  python experiments/analyze_user_distribution.py"
echo "  bash experiments/run_all_analysis.sh"
echo ""
