# apply-fixes.ps1  — harden yfinance fetches (ASCII-only)
param(
  [string]$RepoPath = (Resolve-Path '.').Path,
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }

# Find repo root (handles running from G:\Users\daveq\traderq or inside the repo folder)
if(Test-Path (Join-Path $RepoPath 'app.py')){
  $root = $RepoPath
} elseif(Test-Path (Join-Path $RepoPath 'traderq-sma-tracker\app.py')){
  $root = Join-Path $RepoPath 'traderq-sma-tracker'
} else {
  throw "Could not find app.py in $RepoPath or $RepoPath\traderq-sma-tracker"
}

$venvPy = Join-Path $root '.venv\Scripts\python.exe'
if(-not (Test-Path $venvPy)){ throw "Python venv not found at $venvPy. Run your setup script first." }

# Make sure deps are new enough (no file edits; just install into venv)
Info "Upgrading key packages in venv ..."
& $venvPy -m pip install -U pip wheel
& $venvPy -m pip install -U "yfinance>=0.2.43" "requests>=2.31" "pandas>=2.0" "python-dateutil>=2.8.2" "pytz>=2024.1"

# Write the yfinance patch (ASCII only)
$patchLines = @()
$patchLines += '# yf_patch.py — force a proper session and (optionally) pdr override'
$patchLines += 'import requests'
$patchLines += 'import yfinance as yf'
$patchLines += 'try:'
$patchLines += '    from pandas_datareader import data as pdr  # noqa: F401'
$patchLines += '    yf.pdr_override()'
$patchLines += 'except Exception:'
$patchLines += '    pass'
$patchLines += ''
$patchLines += '# Shared session with realistic headers fixes 403/HTML responses'
$patchLines += 'sess = requests.Session()'
$patchLines += 'sess.headers.update({'
$patchLines += '    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",'
$patchLines += '    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",'
$patchLines += '    "Accept-Language": "en-US,en;q=0.9",'
$patchLines += '    "Connection": "keep-alive"'
$patchLines += '})'
$patchLines += ''
$patchLines += '# yfinance uses shared._base._requests.Session under the hood; set the default session'
$patchLines += 'try:'
$patchLines += '    import yfinance.shared as yfs'
$patchLines += '    yfs._base._requests = sess'
$patchLines += '    yfs._requests = sess'
$patchLines += 'except Exception:'
$patchLines += '    pass'
$patchLines += ''
$patchLines += '# Convenience wrapper if you want to call via yf_patch.download(...)'
$patchLines += 'def download(*args, **kwargs):'
$patchLines += '    kwargs.setdefault("progress", False)'
$patchLines += '    kwargs.setdefault("threads", False)'
$patchLines += '    return yf.download(*args, **kwargs, session=sess)'
$patchPath = Join-Path $root 'yf_patch.py'
$patchLines | Set-Content -Path $patchPath -Encoding Ascii

# Insert 'import yf_patch' at the very top of app.py (only once)
$appPath = Join-Path $root 'app.py'
if(-not (Test-Path $appPath)){ throw "app.py not found at $appPath" }
$lines = Get-Content $appPath -Encoding UTF8
if($lines -notmatch '^\s*import\s+yf_patch'){
  $new = @('import yf_patch  # added by apply-fixes.ps1') + $lines
  $new | Set-Content -Path $appPath -Encoding UTF8
  Info "Inserted 'import yf_patch' at top of app.py"
} else {
  Warn "'import yf_patch' already present; leaving as-is"
}

# Kill any running python/streamlit from this venv (best effort)
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Relaunch app
Push-Location $root
try{
  Info "Starting Streamlit on port $Port ..."
  & $venvPy -m streamlit run app.py --server.port $Port
} finally { Pop-Location }
