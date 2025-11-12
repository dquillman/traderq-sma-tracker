param(
  [string]$Commit = "90c802a",
  [int]$Port = 8630
)

$ErrorActionPreference = "Stop"
Set-Location G:\Users\daveq\traderq

# Repo state
git fetch --all --prune
git checkout -B rescue-ui $Commit
git reset --hard $Commit | Out-Null

# Syntax check (PowerShell-safe, no heredoc)
& python -c "import py_compile; py_compile.compile('app.py', doraise=True)" 

# Venv
if (!(Test-Path .\.venv\Scripts\python.exe)) {
  if (Test-Path 'G:\Python311\python.exe') { & G:\Python311\python.exe -m venv .\.venv } else { & python -m venv .\.venv }
}
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --no-cache-dir streamlit==1.39.0 yfinance==0.2.40 pycoingecko plotly pandas numpy

# Run
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
& python -m streamlit run .\app.py --server.port $Port --server.address localhost
