# find-and-run.ps1
param([int]$Port = 8591)

$ErrorActionPreference = 'Stop'

# 0) Base dir (adjust if you moved the project)
$Base = "G:\Users\daveq\traderq"
Set-Location $Base

# 1) Locate app.py candidates under this folder
$cands = Get-ChildItem -Path $Base -Filter app.py -Recurse -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending

if (!$cands -or $cands.Count -eq 0) {
  Write-Host "❌ No app.py found under $Base" -ForegroundColor Red
  Write-Host "Tip: ensure your repo is checked out and app.py exists at the project root."
  exit 1
}

$App = $cands[0].FullName
Write-Host "✅ Using: $App" -ForegroundColor Green

# 2) Quietly kill stragglers
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name python    -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 300

# 3) Ensure venv
if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
  & "G:\Python311\python.exe" -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

# 4) Deps
if (Test-Path .\requirements.txt) {
  python -m pip install --upgrade pip
  python -m pip install --no-cache-dir -r .\requirements.txt
} else {
  python -m pip install --no-cache-dir streamlit pandas numpy pandas-datareader plotly pycoingecko yfinance
}

# 5) Clear cache
streamlit cache clear | Out-Null

# 6) Run Streamlit with FULL PATH (removes any cwd ambiguity)
Write-Host "Starting Streamlit on http://localhost:$Port ..." -ForegroundColor Cyan
python -m streamlit run "$App" --server.port $Port --server.address localhost
