@echo off
REM ============================================
REM TraderQ Installer Script
REM ============================================
echo.
echo ============================================
echo TraderQ Setup & Installer
echo ============================================
echo.
echo This script will:
echo 1. Check if Python is installed
echo 2. Create a launcher script
echo 3. Set up everything needed to run TraderQ
echo.
pause

REM Change to script directory
cd /d "%~dp0"

REM Check for Python
echo.
echo [1/3] Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ Python is NOT installed!
    echo.
    echo Python is required to run TraderQ.
    echo.
    echo Please install Python:
    echo 1. Go to: https://www.python.org/downloads/
    echo 2. Download Python 3.x (latest version)
    echo 3. During installation, CHECK "Add Python to PATH"
    echo 4. Run this installer again after installing Python
    echo.
    echo Opening Python download page...
    start https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python found! Version: %PYTHON_VERSION%
echo.

REM Check if HTML file exists
echo [2/3] Checking for traderq.html...
if not exist "traderq.html" (
    echo.
    echo ❌ ERROR: traderq.html not found in current directory!
    echo.
    echo Please make sure traderq.html is in the same folder as this installer.
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)
echo ✅ traderq.html found!
echo.

REM Create launcher script
echo [3/3] Creating launcher script...
(
echo @echo off
echo cd /d "%%~dp0"
echo echo ============================================
echo echo Starting TraderQ Server...
echo echo ============================================
echo echo.
echo echo Server will be available at: http://localhost:8000/traderq.html
echo echo.
echo echo IMPORTANT: Keep this window open while using TraderQ!
echo echo Press Ctrl+C to stop the server.
echo echo.
echo echo ============================================
echo echo.
echo python -m http.server 8000
echo if errorlevel 1 (
echo     echo.
echo     echo ERROR: Server failed to start on port 8000!
echo     echo Trying alternative port 8080...
echo     echo.
echo     python -m http.server 8080
echo     if errorlevel 1 (
echo         echo.
echo         echo ERROR: Could not start server on port 8080 either.
echo         echo Please check if Python is installed correctly.
echo         pause
echo         exit /b 1
echo     )
echo     start http://localhost:8080/traderq.html
echo ) else (
echo     start http://localhost:8000/traderq.html
echo )
) > "START_TRADERQ.bat"

echo ✅ Launcher script created: START_TRADERQ.bat
echo.

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo ✅ Setup successful!
echo.
echo To run TraderQ:
echo 1. Double-click "START_TRADERQ.bat" in this folder
echo 2. A server window will open - keep it open
echo 3. Your browser will open automatically, or go to:
echo    http://localhost:8000/traderq.html
echo.
echo ============================================
echo.
echo Would you like to start TraderQ now? (Y/N)
set /p START_NOW="> "
if /i "%START_NOW%"=="Y" (
    echo.
    echo Starting TraderQ...
    start "" "START_TRADERQ.bat"
    timeout /t 2 >nul
    echo.
    echo TraderQ should be opening in your browser!
    echo.
    echo NOTE: The server window must stay open while using TraderQ.
    echo Press Ctrl+C in that window to stop the server when done.
) else (
    echo.
    echo You can start TraderQ anytime by double-clicking "START_TRADERQ.bat"
)
echo.
pause

