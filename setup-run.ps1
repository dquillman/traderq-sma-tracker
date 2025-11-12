# setup-run.ps1 — clean restore + optional glow theme
# ASCII-only, no heredocs. Inserts ui_glow_patch.apply() AFTER set_page_config.
param(
  [string]$RepoUrl = 'https://github.com/dquillman/traderq-sma-tracker',
  [string]$Branch  = 'wip/v1.4.4-broken',
  [string]$FallbackCommit = '90c802a',
  [int]$Port = 8630,
  [switch]$ApplyGlow
)

$ErrorActionPreference = 'Stop'
$cyan='Cyan'; $yellow='Yellow'; $red='Red'
function _info($m){ Write-Host "[INFO] $m" -ForegroundColor $cyan }
function _warn($m){ Write-Host "[WARN] $m" -ForegroundColor $yellow }
function _err ($m){ Write-Host "[ERR ] $m" -ForegroundColor $red }

# Git/Python
if(-not (Get-Command git -ErrorAction SilentlyContinue)){ throw 'git not found on PATH.' }
$pyCandidates = @(
  'G:\Python311\python.exe',
  "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
  "$env:ProgramFiles\Python311\python.exe",
  (Get-Command python -ErrorAction SilentlyContinue | ForEach-Object { $_.Source } | Select-Object -First 1)
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique
if(-not $pyCandidates){ throw 'Python 3.11 not found at G:\Python311\python.exe or on PATH.' }
$pyExe = $pyCandidates[0]

# Repo
$repoName = [IO.Path]::GetFileNameWithoutExtension(($RepoUrl -replace '\.git$',''))
if(-not (Test-Path $repoName)){ _info "Cloning $RepoUrl ..."; git clone $RepoUrl | Out-Null }
$repoPath = (Resolve-Path $repoName).Path

# Reset branch
Push-Location $repoPath
try{
  git fetch --all --prune | Out-Null
  git stash push -u -m 'auto-stash-setup-run' 2>$null | Out-Null
  _info "Checking out $Branch ..."
  git checkout -f $Branch | Out-Null
  if(git branch -r --list "origin/$Branch"){ _info "Hard-reset to origin/$Branch ..."; git reset --hard "origin/$Branch" | Out-Null } else { _warn "origin/$Branch not found; using local branch." }
  git clean -fdx | Out-Null
} finally { Pop-Location }

# Venv
$venvPath = Join-Path $repoPath '.venv'
$venvPy = Join-Path $venvPath 'Scripts\python.exe'
if(-not (Test-Path $venvPy)){ _info "Creating venv at $venvPath ..."; & $pyExe -m venv $venvPath }

# Deps
Push-Location $repoPath
try{
  & $venvPy -m pip install -U pip wheel
  if(Test-Path 'requirements.txt'){ _info 'Installing requirements.txt ...'; & $venvPy -m pip install -r requirements.txt } else { _warn 'requirements.txt not found; skipping.' }
} finally { Pop-Location }

# Syntax
Push-Location $repoPath
try{ _info 'Compiling Python files (syntax check) ...'; & $venvPy -m compileall -q . } finally { Pop-Location }

# Smoke test
Push-Location $repoPath
$smoke=$false; $p=$null
try{
  $args=@('-m','streamlit','run','app.py','--server.port',$Port,'--server.headless','true')
  _info "Smoke test: start Streamlit on port $Port ..."
  $p=Start-Process -FilePath $venvPy -ArgumentList $args -PassThru -WindowStyle Hidden
  Start-Sleep -Seconds 6
  if($p -and -not $p.HasExited){ _info 'Smoke test OK (process alive).'; $smoke=$true } else { $code=if($p){$p.ExitCode}else{'n/a'}; _err "Streamlit exited early (code $code)." }
} finally { if($p -and -not $p.HasExited){ Stop-Process -Id $p.Id -Force }; Pop-Location }

# Fallback
if(-not $smoke){
  Push-Location $repoPath
  try{
    _warn "Falling back to commit $FallbackCommit ..."
    git checkout -f $FallbackCommit | Out-Null
    git clean -fdx | Out-Null
    & $venvPy -m pip install -U pip wheel
    if(Test-Path 'requirements.txt'){ & $venvPy -m pip install -r requirements.txt }
    & $venvPy -m compileall -q .
  } finally { Pop-Location }
}

# Optional glow
if($ApplyGlow){
  _info 'Applying glow UI (ASCII-only, delayed injection) ...'
  $stylesDir = Join-Path $repoPath 'styles'
  if(-not (Test-Path $stylesDir)){ New-Item -ItemType Directory -Path $stylesDir | Out-Null }

  $css = @(
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
  )
  $css | Set-Content -Path (Join-Path $stylesDir 'glow.css') -Encoding Ascii

  # Injector: define apply(), do NOT auto-run on import
  $patch = @(
    '# ui_glow_patch.py : injects styles/glow.css after set_page_config (ASCII only)',
    'from pathlib import Path',
    'import streamlit as st',
    'def apply():',
    '    css_path = Path(__file__).resolve().parent.joinpath("styles", "glow.css")',
    '    if css_path.exists():',
    '        css = css_path.read_text(encoding="ascii", errors="ignore")',
    '        st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)'
  )
  $patch | Set-Content -Path (Join-Path $repoPath 'ui_glow_patch.py') -Encoding Ascii

  # Ensure import present (safe anywhere, no Streamlit call on import)
  $appPath = Join-Path $repoPath 'app.py'
  if(-not (Test-Path $appPath)){ throw "app.py not found at $repoPath" }
  $lines = Get-Content $appPath -Encoding UTF8
  $changed = $false

  if($lines -notmatch 'import ui_glow_patch'){
    # Insert after 'import streamlit' if found, else top
    $new = @(); $inserted = $false
    foreach($ln in $lines){
      if(-not $inserted -and $ln -match 'import\s+streamlit'){
        $new += $ln
        $new += 'import ui_glow_patch  # added by setup-run.ps1'
        $inserted = $true
        $changed = $true
      } else { $new += $ln }
    }
    if(-not $inserted){ $new = @('import ui_glow_patch  # added by setup-run.ps1') + $lines; $changed = $true }
    $lines = $new
  }

  # Insert a call *after* first set_page_config(...) and only if missing
  if($lines -notmatch 'ui_glow_patch\.apply\(\)'){
    $new2 = @(); $applied = $false
    for($i=0; $i -lt $lines.Count; $i++){
      $new2 += $lines[$i]
      if(-not $applied -and $lines[$i] -match 'st\.set_page_config\s*\('){
        $new2 += 'ui_glow_patch.apply()  # added by setup-run.ps1'
        $applied = $true
        $changed = $true
      }
    }
    if(-not $applied){
      # Fallback: append at end; better than injecting before SPC
      $new2 += ''
      $new2 += '# setup-run.ps1 fallback: apply glow at end if set_page_config not found'
      $new2 += 'ui_glow_patch.apply()'
      $changed = $true
    }
    $lines = $new2
  }

  if($changed){ $lines | Set-Content -Path $appPath -Encoding UTF8; _info 'Glow patch wired after set_page_config.' } else { _info 'Glow patch already wired; no changes.' }
}

# Run foreground
Push-Location $repoPath
try{
  _info "Starting Streamlit on port $Port ..."
  & $venvPy -m streamlit run app.py --server.port $Port
} finally { Pop-Location }
