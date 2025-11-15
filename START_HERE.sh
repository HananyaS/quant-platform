#!/bin/bash
echo ""
echo ""
echo "    ================================================"
echo "    ║                                              ║"
echo "    ║     ✅ SETUP COMPLETE!                       ║"
echo "    ║                                              ║"
echo "    ║     Launching Trading Platform...            ║"
echo "    ║                                              ║"
echo "    ║     Your browser will open automatically     ║"
echo "    ║                                              ║"
echo "    ================================================"
echo ""
echo ""

# Launch the platform
python -m streamlit run app.py

echo ""
echo "Platform closed. Press any key to exit..."
read -n 1 -s

