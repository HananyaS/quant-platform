#!/bin/bash

clear
echo ""
echo ""
echo "    ================================================"
echo "    ║                                              ║"
echo "    ║     QUANTITATIVE TRADING PLATFORM            ║"
echo "    ║                                              ║"
echo "    ║     🚀 ONE-CLICK LAUNCH                      ║"
echo "    ║                                              ║"
echo "    ================================================"
echo ""
echo ""
echo "    Setting up and launching platform..."
echo ""
echo "    Please wait while we:"
echo "      1. Check dependencies"
echo "      2. Install if needed"
echo "      3. Launch platform"
echo ""
echo "    This may take 1-2 minutes on first run."
echo ""
echo ""

# Activate virtual environment if it exists
if [ -f "../../../venvs/fin_venv/bin/activate" ]; then
    source ../../../venvs/fin_venv/bin/activate 2>/dev/null
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate 2>/dev/null
fi

# Install dependencies silently
pip install --quiet --upgrade pip 2>/dev/null
pip install --quiet pandas numpy scipy matplotlib seaborn plotly streamlit PyYAML yfinance scikit-learn 2>/dev/null

clear
echo ""
echo ""
echo "    ================================================"
echo "    ║                                              ║"
echo "    ║     ✅ SETUP COMPLETE!                       ║"
echo "    ║                                              ║"
echo "    ║     Launching Trading Platform...            ║"
echo "    ║                                              ║"
echo "    ║     Your browser will open automatically    ║"
echo "    ║                                              ║"
echo "    ================================================"
echo ""
echo ""

# Launch the platform
python -m streamlit run app.py

echo ""
echo "Platform closed. Press any key to exit..."
read -n 1 -s

