@echo off
setlocal enabledelayedexpansion

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

REM Kill any existing Streamlit processes on port 8501
echo Checking for existing processes on port 8501...
set "found_process=0"
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8501 ^| findstr LISTENING') do (
    echo Killing existing process on port 8501 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
    set "found_process=1"
)

if !found_process!==1 (
    echo Waiting for port to be released...
    timeout /t 3 /nobreak >nul
) else (
    echo No existing processes found on port 8501.
)

echo.
echo Starting Streamlit...
echo ========================================
echo.
echo The app will open in your default browser at:
echo   http://localhost:8501
echo.
echo To stop the server:
echo   - Close this window, OR
echo   - Press Ctrl+C in this window
echo.
echo ========================================
echo.

REM Run Streamlit directly - this will block and keep the window open
.venv\Scripts\python.exe -m streamlit run app.py --server.port=8501

REM If we get here, Streamlit has exited
echo.
echo.
echo TraderQ has stopped.
echo Cleaning up any remaining processes...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo.
pause
