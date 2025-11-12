# flatten-repo.ps1 — promote inner repo to parent, keep history, single venv, run app on 8630
param(
  [string]$Root = "G:\Users\daveq\traderq",
  [string]$InnerName = "traderq-sma-tracker",
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Errx($m){ Write-Host "[ERR ] $m" -ForegroundColor Red }

# 0) Preconditions
if(-not (Test-Path $Root)){ throw "Root not found: $Root" }
$Inner = Join-Path $Root $InnerName
if(-not (Test-Path (Join-Path $Inner ".git"))){ throw "No .git in $Inner — are you sure this is the cloned repo?" }

# 1) Stop any running processes
Get-Process -Name streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# 2) Backup loose parent files (anything that’s NOT the inner folder) to a timestamped backup
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$Backup = Join-Path $Root ("_backup_before_flatten_" + $ts)
New-Item -ItemType Directory -Path $Backup | Out-Null
Info "Backing up loose parent files to $Backup ..."
Get-ChildItem $Root -Force | Where-Object { $_.Name -ne $InnerName -and $_.Name -ne '.' -and $_.Name -ne '..' } | `
  ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $Backup }

# 3) Move repo contents (including .git) up into $Root
Info "Promoting repo $InnerName to $Root ..."
# Use robocopy to move everything including hidden; then remove empty dir
$rc = robocopy $Inner $Root /E /MOVE /R:1 /W:1 /NFL /NDL /NP /NJH /NJS
# robocopy exit codes: 0/1 OK; >7 failure
if($LASTEXITCODE -gt 7){ throw "robocopy failed with code $LASTEXITCODE" }
# If folder still remains, try remove
if(Test-Path $Inner){ Remove-Item $Inner -Recurse -Force -ErrorAction SilentlyContinue }

# 4) Restore useful parent files from backup into the now-promoted repo root
# (setup-run.ps1, apply-fixes.ps1, any others you care about)
Get-ChildItem $Backup -Force | ForEach-Object {
  $dest = Join-Path $Root $_.Name
  if(-not (Test-Path $dest)){
    Move-Item -LiteralPath $_.FullName -Destination $dest
  } else {
    Warn "Skipping restore of '$($_.Name)' (already exists after promote). Check $Backup if needed."
  }
}

# 5) Handle venv: prefer an existing .venv at root; else if backup had one, restore that; else keep moved one
$rootVenv = Join-Path $Root ".venv"
if(-not (Test-Path (Join-Path $rootVenv "Scripts\python.exe"))){
  # See if backup had a venv
  $bkpVenv = Join-Path $Backup ".venv"
  if(Test-Path (Join-Path $bkpVenv "Scripts\python.exe")){
    Info "Restoring .venv from backup ..."
    Move-Item -LiteralPath $bkpVenv -Destination $rootVenv
  }
}

# 6) Ensure patch files exist (ASCII-only); recreate if missing
if(-not (Test-Path (Join-Path $Root "ui_glow_patch.py"))){
  Info "Recreating ui_glow_patch.py ..."
  @(
    '# ui_glow_patch.py : injects styles/glow.css after set_page_config (ASCII only)',
    'from pathlib import Path',
    'import streamlit as st',
    'def apply():',
    '    css_path = Path(__file__).resolve().parent.joinpath("styles", "glow.css")',
    '    if css_path.exists():',
    '        css = css_path.read_text(encoding="ascii", errors="ignore")',
    '        st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)'
  ) | Set-Content -Path (Join-Path $Root "ui_glow_patch.py") -Encoding Ascii
}
$styles = Join-Path $Root "styles"
if(-not (Test-Path $styles)){ New-Item -ItemType Directory -Path $styles | Out-Null }
if(-not (Test-Path (Join-Path $styles "glow.css"))){
  Info "Recreating styles\glow.css ..."
  @(
    '/* glow.css : ASCII-only neon-style theme for Streamlit */',
    'html, body, .stApp {',
    '  background-color: #0a0f1a;',
    '  color: #d7e1ff;',
    '}',
    '.stApp {',
    '  text-shadow: 0 0 2px #00e5ff;',
    '}',
    'h1, h2, h3 {',
    '  color: #a7c7ff;',
    '  text-shadow: 0 0 4px #33ccff, 0 0 8px #33ccff;',
    '}',
    'div[data-baseweb="tab"] {',
    '  border-bottom: 1px solid #123;',
    '}',
    '.stButton>button, .stDownloadButton>button {',
    '  background: #0d1b2a;',
    '  border: 1px solid #1b3555;',
    '  box-shadow: 0 0 6px #33ccff;',
    '  color: #d7e1ff;',
    '}',
    '.stButton>button:hover, .stDownloadButton>button:hover {',
    '  box-shadow: 0 0 8px #66e0ff, 0 0 12px #66e0ff;',
    '  border-color: #33ccff;',
    '}',
    '.stTextInput input, .stSelectbox, .stNumberInput input {',
    '  background: #0d1b2a;',
    '  color: #d7e1ff;',
    '  border: 1px solid #1b3555;',
    '  box-shadow: inset 0 0 4px #10263d;',
    '}',
    '.stDataFrame, .stTable {',
    '  filter: drop-shadow(0 0 6px #123);',
    '}',
    'div[data-testid="stHorizontalBlock"] > div {',
    '  border-radius: 6px;',
    '  box-shadow: 0 0 10px #0b2a40;',
    '}',
    'label[data-baseweb="checkbox"] {',
    '  filter: drop-shadow(0 0 4px #33ccff);',
    '}'
  ) | Set-Content -Path (Join-Path $styles "glow.css") -Encoding Ascii
}

# 7) Make sure yf_patch.py is present
if(-not (Test-Path (Join-Path $Root "yf_patch.py"))){
  Info "Recreating yf_patch.py ..."
  @(
    '# yf_patch.py — force a proper session and (optionally) pdr override',
    'import requests',
    'import yfinance as yf',
    'try:',
    '    from pandas_datareader import data as pdr  # noqa: F401',
    '    yf.pdr_override()',
    'except Exception:',
    '    pass',
    '',
    'sess = requests.Session()',
    'sess.headers.update({',
    '    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",',
    '    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",',
    '    "Accept-Language": "en-US,en;q=0.9",',
    '    "Connection": "keep-alive"',
    '})',
    'try:',
    '    import yfinance.shared as yfs',
    '    yfs._base._requests = sess',
    '    yfs._requests = sess',
    'except Exception:',
    '    pass',
    '',
    'def download(*args, **kwargs):',
    '    kwargs.setdefault("progress", False)',
    '    kwargs.setdefault("threads", False)',
    '    return yf.download(*args, **kwargs, session=sess)'
  ) | Set-Content -Path (Join-Path $Root "yf_patch.py") -Encoding Ascii
}

# 8) Ensure app imports are sane (future import first, then streamlit, then yf_patch)
$appPath = Join-Path $Root "app.py"
if(-not (Test-Path $appPath)){ throw "app.py not found at $appPath" }
$lines = Get-Content $appPath -Encoding UTF8
# Remove dup 'import yf_patch'
$lines = $lines | Where-Object { $_ -notmatch '^\s*import\s+yf_patch\b' }
# Ensure future import exists and is at top (after comments)
$lines = $lines | Where-Object { $_ -notmatch '^\s*from\s+__future__\s+import\s+annotations\s*$' }
$idx = 0; while($idx -lt $lines.Count -and $lines[$idx] -match '^\s*#'){ $idx++ }
$before = if($idx -gt 0){ $lines[0..($idx-1)] } else { @() }
$after  = $lines[$idx..($lines.Count-1)]
$lines = @(); $lines += $before; $lines += 'from __future__ import annotations'; $lines += $after
# Insert streamlit then yf_patch
$stPos = ($lines | Select-String -Pattern '^\s*import\s+streamlit\s+as\s+st\s*$').LineNumber
if($stPos){
  $i = [int]$stPos[0] - 1
  $pre = $lines[0..$i]
  $post = if($i+1 -lt $lines.Count){ $lines[($i+1)..($lines.Count-1)] } else { @() }
  $lines = @(); $lines += $pre; $lines += 'import yf_patch  # added by flatten-repo.ps1'; $lines += $post
} else {
  # fallback: put after future import line
  $fiPos = ($lines | Select-String -Pattern '^\s*from\s+__future__\s+import\s+annotations\s*$').LineNumber
  $i = if($fiPos){ [int]$fiPos[0] } else { 0 }
  $pre = if($i -gt 0){ $lines[0..($i-1)] } else { @() }
  $post = $lines[$i..($lines.Count-1)]
  $lines = @(); $lines += $pre; $lines += $post[0]; $lines += 'import yf_patch  # added by flatten-repo.ps1'; if($post.Count -gt 1){ $lines += $post[1..($post.Count-1)] }
}
$lines | Set-Content -Path $appPath -Encoding UTF8

# 9) Ensure venv exists
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if(-not (Test-Path $venvPy)){
  Info "Creating venv at $Root\.venv ..."
  & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv (Join-Path $Root ".venv")
  $venvPy = Join-Path $Root ".venv\Scripts\python.exe"
  if(-not (Test-Path $venvPy)){ throw "Could not create venv at $Root\.venv" }
}

# 10) Install deps + smoke test + run
Push-Location $Root
try{
  & $venvPy -m pip install -U pip wheel
  if(Test-Path 'requirements.txt'){ & $venvPy -m pip install -r requirements.txt }
  & $venvPy -m compileall -q .
  Info "Starting Streamlit on port $Port ..."
  & $venvPy -m streamlit run app.py --server.port $Port
} finally { Pop-Location }
