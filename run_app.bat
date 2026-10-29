@echo off
echo ======================================================================
echo Launching Capstone 16 - Responsible AI and Bias Audit Application
echo ======================================================================
cd /d "%~dp0"

if exist "..\.venv\Scripts\streamlit.exe" (
    echo [OK] Using virtual environment at ..\.venv
    "..\.venv\Scripts\streamlit.exe" run app.py --browser.gatherUsageStats=false
) else if exist ".venv\Scripts\streamlit.exe" (
    echo [OK] Using virtual environment at .venv
    ".venv\Scripts\streamlit.exe" run app.py --browser.gatherUsageStats=false
) else (
    echo [OK] Using system python streamlit
    python -m streamlit run app.py --browser.gatherUsageStats=false
)
pause
