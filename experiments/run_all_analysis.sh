#!/bin/bash

# Run all analysis scripts
# Usage: source venv/bin/activate && bash experiments/run_all_analysis.sh

echo "===================================================================="
echo "RUNNING ALL ANALYSIS SCRIPTS"
echo "===================================================================="

# Activate venv if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo ""
    echo "⚠️  Virtual environment not activated!"
    echo "   Run: source venv/bin/activate"
    echo ""
    exit 1
fi

echo ""
echo "[1/5] User Distribution Analysis"
echo "--------------------------------------------------------------------"
python experiments/analyze_user_distribution.py

echo ""
echo "[2/5] Quick Results Comparison"
echo "--------------------------------------------------------------------"
python experiments/quick_compare.py

echo ""
echo "[3/5] Detailed Results Analysis"
echo "--------------------------------------------------------------------"
python experiments/analyze_results.py

echo ""
echo "[4/5] Statistical Significance Tests"
echo "--------------------------------------------------------------------"
python experiments/statistical_tests.py

echo ""
echo "[5/5] Creating Visualizations"
echo "--------------------------------------------------------------------"
python experiments/create_visualizations.py

echo ""
echo "===================================================================="
echo "✅ ALL ANALYSIS COMPLETE"
echo "===================================================================="
echo ""
echo "📊 Check results in:"
echo "   - Terminal output above"
echo "   - data/graphs/ for visualizations"
echo ""
