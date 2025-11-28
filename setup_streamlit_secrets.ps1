# Setup Streamlit Secrets - Helper Script
# This script will help you copy your Firebase secrets to Streamlit Cloud

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Streamlit Secrets Setup Helper" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if TOML file exists
$tomlFile = ".streamlit_secrets_toml.txt"
if (-not (Test-Path $tomlFile)) {
    Write-Host "❌ TOML file not found. Generating it now..." -ForegroundColor Yellow
    python convert_key_to_toml.py
    if (-not (Test-Path $tomlFile)) {
        Write-Host "❌ Failed to generate TOML file. Please run: python convert_key_to_toml.py" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Found TOML configuration file!" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  STEP 1: Copy Secrets Content" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Opening file for you to copy..." -ForegroundColor Yellow
Start-Process notepad.exe $tomlFile
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  STEP 2: Go to Streamlit Cloud" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Open this URL in your browser:" -ForegroundColor White
Write-Host "   https://share.streamlit.io" -ForegroundColor Green
Write-Host ""
Write-Host "2. Select your app: traderq-sma-tracker" -ForegroundColor White
Write-Host ""
Write-Host "3. Go to: Settings → Secrets" -ForegroundColor White
Write-Host ""
Write-Host "4. Copy ALL content from the notepad window" -ForegroundColor White
Write-Host ""
Write-Host "5. Paste into the Secrets text box" -ForegroundColor White
Write-Host ""
Write-Host "6. Click 'Save'" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$response = Read-Host "Would you like me to open Streamlit Cloud in your browser? (Y/N)"
if ($response -eq "Y" -or $response -eq "y") {
    Write-Host "Opening Streamlit Cloud..." -ForegroundColor Green
    Start-Process "https://share.streamlit.io"
}

Write-Host ""
Write-Host "✅ Setup instructions complete!" -ForegroundColor Green
Write-Host "   After saving secrets, wait 2-3 minutes for the app to redeploy." -ForegroundColor Yellow

