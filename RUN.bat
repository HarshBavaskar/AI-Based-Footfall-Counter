@echo off
cls
echo ========================================
echo   AI Footfall Counter - Made by Harsh Bavaskar
echo   Models used: Yolo v8n, DeepSort
echo ========================================
echo.
if not exist venv python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python gui.py
