# setup-traderq.ps1
# Cleans C: user Python, pins PATH to G:\Python311, builds .venv, installs deps, runs Streamlit app

param([switch]$NoRun)

function Info($m){ Write-Host "[i] $m" }
function OK($m){ Write-Host "[OK] $m" }
function Warn($m){ Write-Warning $m }
function Die($m){ Write-Error $m; exit 1 }

# Move to script folder
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Info "Working dir: $ScriptDir"

# Stop stuck installers / python (ignore errors)
Get-Process winget, msiexec, python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Remove C: per-user Python (if present)
$CPython = "$env:LOCALAPPDATA\Programs\Python"
if (Test-Path $CPython) {
  Info "Removing $CPython ..."
  Remove-Item -Recurse -Force $CPython -ErrorAction SilentlyContinue
  if (Test-Path $CPython) { Warn "Some files locked. Retry after reboot if needed." } else { OK "Removed C: user Python." }
} else {
  Info "No per-user Python under $CPython (good)."
}

# Remove WindowsApps shims if present
$WinApps = "$env:LOCALAPPDATA\Microsoft\WindowsApps"
@("Python.exe","python3.exe","py.exe","py3.exe") | ForEach-Object {
  $p = Join-Path $WinApps $_
  if (Test-Path $p) {
    Info "Removing shim $p"
    Remove-Item -Recurse -Force $p -ErrorAction SilentlyContinue
  }
}

# Ensure G:\Python311 exists
$GPython = "G:\Python311\python.exe"
if (-not (Test-Path $GPython)) {
  Die "Python 3.11 expected at $GPython but not found. Install Python 3.11 x64 to G:\Python311, then re-run."
}

# Update USER PATH to prefer G:\Python311
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$parts = @()
if ($UserPath) { $parts = $UserPath -split ';' | Where-Object { $_ -and $_.Trim() -ne "" } }

# remove C:\Users\...\Programs\Python entries
$parts = $parts | Where-Object { $_ -notmatch "\\Users\\[^\\]+\\AppData\\Local\\Programs\\Python" }

# ensure G:\Python311 and Scripts are present
$gRoot = "G:\Python311\"
$gScripts = "G:\Python311\Scripts\"
if (-not ($parts -contains $gRoot))   { $parts += $gRoot }
if (-not ($parts -contains $gScripts)){ $parts += $gScripts }

$NewUserPath = ($parts | Select-Object -Unique) -join ';'
[Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
OK "Updated User PATH for this account."

# Also update this session's PATH so we don't need a new shell
$env:Path = $NewUserPath + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")

# Rebuild project .venv with G:\Python311
if (Test-Path ".venv") {
  Info "Removing existing .venv ..."
  Remove-Item -Recurse -Force ".venv" -ErrorAction SilentlyContinue
}
Info "Creating .venv with $GPython ..."
& $GPython -m venv .venv | Out-Null
$VenvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { Die "Failed to create .venv with $GPython" }
OK ".venv created."

# Upgrade toolchain
& $VenvPy -m pip install --upgrade pip wheel setuptools

# Ensure requirements.txt
if (-not (Test-Path "requirements.txt")) {
@"
streamlit==1.39.0
yfinance==0.2.44
pandas==2.2.2
numpy==1.26.4
plotly==5.24.1
"@ | Out-File -Encoding UTF8 "requirements.txt"
  OK "Created requirements.txt"
}

# Install requirements
Info "Installing requirements (no cache) ..."
& $VenvPy -m pip install --no-cache-dir -r requirements.txt
if ($LASTEXITCODE -ne 0) { Die "Package install failed. Check network/antivirus and re-run." }

# Sanity imports
Info "Verifying imports ..."
& $VenvPy -c "import yfinance as yf, pandas as pd, numpy as np; print('OK yfinance', yf.__version__); print('OK pandas', pd.__version__); print('OK numpy', np.__version__)"
if ($LASTEXITCODE -ne 0) { Die "Import check failed inside venv." }
OK "Imports OK."

# Launch app unless -NoRun
if (-not $NoRun) {
  Info "Launching app at http://localhost:8501 ..."
  & $VenvPy -m streamlit run app.py
} else {
  OK "Setup complete. Not launching because -NoRun specified."
}
