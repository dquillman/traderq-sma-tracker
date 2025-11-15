param(
  [int]$Port = 8605,
  [string]$Message = "",
  [switch]$Minor,
  [switch]$Major,
  [switch]$NoRun
)

$ErrorActionPreference = "Stop"
Set-Location G:\Users\daveq\traderq

function OK($m){ Write-Host "[OK] $m" -ForegroundColor Green }
function Info($m){ Write-Host "[i] $m" -ForegroundColor Cyan }
function Die($m){ Write-Error $m; exit 1 }

# --- 1) Bump APP_VERSION in app.py ---
if (!(Test-Path .\app.py)) { Die "app.py not found." }
$text = Get-Content .\app.py -Raw -Encoding UTF8
$verMatch = [regex]::Match($text, 'APP_VERSION\s*=\s*"(v?\d+\.\d+\.\d+)"')
if (-not $verMatch.Success) { Die "APP_VERSION not found in app.py" }
$oldVer = $verMatch.Groups[1].Value.TrimStart('v')
$parts = $oldVer.Split('.').ForEach({ [int]$_ })
$major,$minor,$patch = $parts[0],$parts[1],$parts[2]

if ($Major) { $major++; $minor=0; $patch=0 }
elseif ($Minor) { $minor++; $patch=0 }
else { $patch++ }

$newVer = "v{0}.{1}.{2}" -f $major,$minor,$patch
$text = [regex]::Replace($text, 'APP_VERSION\s*=\s*"(?:[^"]*)"', "APP_VERSION = `"$newVer`"", 1)
Set-Content .\app.py $text -Encoding UTF8
OK "Version bumped: $($oldVer) -> $newVer"

# --- 2) Git add/commit/push ---
& git status | Out-Null
& git add -A
if ([string]::IsNullOrWhiteSpace($Message)) {
  $Message = "chore: sync $newVer"
}
& git commit -m $Message 2>$null | Out-Null
# If nothing to commit, the previous command exits non-zero – ignore harmlessly
$remote = (& git remote -v) -join "`n"
if ($remote -notmatch 'origin\s+https?://github.com/.+\.git') {
  Die "Git remote 'origin' not set to GitHub HTTPS. Current remotes:`n$remote"
}
OK "Commit created (or no changes). Pushing..."
& git push origin main
OK "Pushed to origin/main."

if ($NoRun) { Info "Skipping run due to -NoRun"; exit 0 }

# --- 3) Prep venv / deps ---
$python = ".\.venv\Scripts\python.exe"
if (!(Test-Path $python)) {
  Info "Creating virtual environment..."
  # Prefer G:\Python311\python.exe if present, else fallback to PATH python
  $pyBase = "G:\Python311\python.exe"
  if (Test-Path $pyBase) {
    & $pyBase -m venv .\.venv
  } else {
    & python -m venv .\.venv
  }
}
. .\.venv\Scripts\Activate.ps1
OK "Venv ready: $((.\.venv\Scripts\python.exe -V))"

Info "Installing/refreshing deps..."
python -m pip install --upgrade pip >$null
python -m pip install --no-cache-dir -U streamlit==1.39.0 yfinance pycoingecko plotly pandas numpy pandas-datareader >$null
OK "Dependencies installed."

# --- 4) Kill anything bound to $Port and start Streamlit ---
try {
  Get-NetTCPConnection -LocalPort $Port -ErrorAction Stop | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
  }
} catch {}

$log = Join-Path $PWD "streamlit_log.txt"
Remove-Item $log -ErrorAction SilentlyContinue

Info "Launching Streamlit on http://localhost:$Port ..."
$env:STREAMLIT_SERVER_ENABLECORS = "true"
$env:STREAMLIT_SERVER_ENABLEXSRFPROTECTION = "true"
$proc = Start-Process -FilePath python `
  -ArgumentList @("-m","streamlit","run","app.py","--server.port",$Port,"--server.address","localhost") `
  -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $log

Start-Sleep -Seconds 2
OK "Started (PID $($proc.Id)). Log: $log"
Write-Host "Open: http://localhost:$Port"
