param(
  [string]$Commit = "90c802a",      # <- put your known-good commit here
  [string]$BaseBranch = "clean-base",
  [string]$RunBranch = "rescue-ui",
  [switch]$Force,                   # allow hard reset even if working tree is dirty
  [switch]$Run,                     # launch Streamlit after setup
  [int]$Port = 8630
)

$ErrorActionPreference = "Stop"
Set-Location G:\Users\daveq\traderq

function OK($m){ Write-Host "[OK] $m" -ForegroundColor Green }
function INFO($m){ Write-Host "[i]  $m" -ForegroundColor Cyan }
function DIE($m){ Write-Error $m; exit 1 }

# --- Sanity checks
git rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) { DIE "Not a git repo here." }

# refuse to clobber uncommitted work unless -Force
$dirty = (git status --porcelain).Trim()
if ($dirty -and -not $Force) {
  DIE "Working tree is dirty. Commit/stash or re-run with -Force."
}

# --- Fetch and reset to good commit on a fresh branch
INFO "Fetching..."
git fetch --all --prune

INFO "Checking out $RunBranch from origin/main (fallback to main)..."
git checkout -B $RunBranch origin/main 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { git checkout -B $RunBranch main | Out-Null }
OK "On $RunBranch"

INFO "Hard reset to $Commit ..."
git reset --hard $Commit | Out-Null
OK "Repo state set to $Commit"

# --- Create and push the clean base branch from this known-good state
INFO "Creating $BaseBranch at $Commit ..."
git branch -f $BaseBranch $Commit | Out-Null
git checkout $BaseBranch | Out-Null
git push -u origin $BaseBranch -f | Out-Null
OK "Pushed $BaseBranch to origin"

# Switch back to run branch (same state) so you can iterate there
git checkout $RunBranch | Out-Null

# --- Add pre-commit hook to block non-ASCII in common text/code files
$hooks = Join-Path (git rev-parse --git-dir).Trim() "hooks"
if (!(Test-Path $hooks)) { New-Item -ItemType Directory -Path $hooks | Out-Null }

# PowerShell hook worker
$psHook = @'
# .git/hooks/pre-commit.ps1  (ASCII-only)
$ErrorActionPreference = "Stop"
$files = git diff --cached --name-only | Where-Object {
  $_ -match '\.(py|toml|md|css)$'
}
$bad = @()
foreach ($f in $files) {
  if (Test-Path $f) {
    $bytes = [System.IO.File]::ReadAllBytes($f)
    foreach ($b in $bytes) {
      if ($b -gt 127) { $bad += $f; break }
    }
  }
}
if ($bad.Count -gt 0) {
  Write-Host "Commit blocked: non-ASCII bytes found in:" -ForegroundColor Red
  $bad | Sort-Object -Unique | ForEach-Object { Write-Host " - $_" }
  Write-Host "Tip: replace typographic quotes/em-dashes/emoji with ASCII." -ForegroundColor Yellow
  exit 1
}
exit 0
'@

# Bash launcher that invokes PowerShell hook (Git runs this file)
$bashLauncher = @'
#!/usr/bin/env bash
powershell -NoProfile -ExecutionPolicy Bypass -File "$(git rev-parse --git-dir)/hooks/pre-commit.ps1"
exit $?
'@

Set-Content -LiteralPath (Join-Path $hooks "pre-commit.ps1") -Value $psHook -Encoding ASCII
Set-Content -LiteralPath (Join-Path $hooks "pre-commit") -Value $bashLauncher -Encoding ASCII
# Try to set exec bit (harmless on Windows)
try { & git update-index --chmod=+x .git/hooks/pre-commit 2>$null } catch {}

OK "Pre-commit hook installed (blocks non-ASCII in .py/.toml/.md/.css)."

# --- Verify Python compiles before we run
if (Test-Path .\app.py) {
  INFO "Syntax check: app.py"
  try {
    python - << 'PY'
import py_compile, sys
py_compile.compile("app.py", doraise=True)
print("OK: app.py compiles")
PY
  } catch {
    DIE "Python syntax error. Fix app.py then commit."
  }
} else {
  INFO "No app.py at this commit."
}

# --- Optional: run Streamlit on chosen port
if ($Run) {
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  # Venv (optional but helpful)
  if (!(Test-Path .\.venv\Scripts\python.exe)) {
    if (Test-Path "G:\Python311\python.exe") { & G:\Python311\python.exe -m venv .\.venv } else { & python -m venv .\.venv }
  }
  . .\.venv\Scripts\Activate.ps1 2>$null | Out-Null
  python -m pip install --upgrade pip *> $null
  python -m pip install --no-cache-dir streamlit==1.39.0 yfinance==0.2.40 pycoingecko plotly pandas numpy *> $null
  Write-Host "Launching http://localhost:$Port"
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}

OK "Done. Branches: $BaseBranch (safe), $RunBranch (active)."
