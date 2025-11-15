# Simple script to run TraderQ
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting TraderQ SMA Tracker" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "ERROR: app.py not found!" -ForegroundColor Red
    exit 1
}

# Try to use venv Python first, fallback to system Python
$python = $null
if (Test-Path ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
    Write-Host "Using virtual environment Python" -ForegroundColor Green
} elseif (Test-Path "G:\Python311\python.exe") {
    $python = "G:\Python311\python.exe"
    Write-Host "Using system Python from G:\Python311" -ForegroundColor Yellow
} else {
    $python = "python"
    Write-Host "Using system Python from PATH" -ForegroundColor Yellow
}

# Check if streamlit is installed
Write-Host "Checking for Streamlit..." -ForegroundColor Cyan
$streamlitCheck = & $python -m streamlit --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
    & $python -m pip install -q -r requirements.txt
}

# Kill any existing processes on port 8501
Write-Host "Checking for existing processes on port 8501..." -ForegroundColor Cyan
$processes = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($processes) {
    foreach ($pid in $processes) {
        Write-Host "Killing process on port 8501 (PID: $pid)" -ForegroundColor Yellow
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "Starting Streamlit on http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Run Streamlit
& $python -m streamlit run app.py --server.port 8501

