
@echo off
title Local AI Assistant V2

cd /d "C:\Users\hp\Downloads\local_ai_assistant_v2\local_ai_v2"

call .venv\Scripts\activate

start "" http://localhost:8502

python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8502

pause
