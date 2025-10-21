#!/bin/bash
clear
echo "========================================"
echo "  AI Footfall Counter"
echo "  Made by HARSH BAVASKAR"
echo "========================================"
echo ""
if [ ! -d "venv" ]; then python3 -m venv venv; fi
source venv/bin/activate
pip install -r requirements.txt
python gui.py
