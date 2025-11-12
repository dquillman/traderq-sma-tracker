@echo off
echo ========================================
echo   Starting TraderQ SMA Tracker
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first or check that .venv folder exists.
    pause
    exit /b 1
)

REM Activate virtual environment and run Streamlit
echo Starting Streamlit app...
echo.
echo The app will open in your default browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

.venv\Scripts\python.exe -m streamlit run app.py --server.port=8501

pause
