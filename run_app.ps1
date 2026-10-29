# Run Streamlit Application in PowerShell
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Launching Capstone 16 - Responsible AI and Bias Audit Application" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot

if (Test-Path "..\.venv\Scripts\streamlit.exe") {
    Write-Host "[OK] Using virtual environment at ..\.venv" -ForegroundColor Yellow
    & "..\.venv\Scripts\streamlit.exe" run app.py --browser.gatherUsageStats=false
} elseif (Test-Path ".\.venv\Scripts\streamlit.exe") {
    Write-Host "[OK] Using virtual environment at .\.venv" -ForegroundColor Yellow
    & ".\.venv\Scripts\streamlit.exe" run app.py --browser.gatherUsageStats=false
} else {
    Write-Host "[OK] Using system streamlit" -ForegroundColor Yellow
    python -m streamlit run app.py --browser.gatherUsageStats=false
}
