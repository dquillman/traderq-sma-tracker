# fix-encoding.ps1 — remove mis-encoded characters and restore clean header
param([int]$Port = 8617)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

$src = 'app.py'
if (!(Test-Path $src)) { throw "app.py not found" }

# Backup
$bak = "$src.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $src $bak -Force
Write-Host "[OK] Backup -> $bak"

# Read and clean
$txt = Get-Content $src -Raw -Encoding UTF8
# Replace any mis-encoded characters like âš¡ or â€” etc.
$txt = $txt -replace 'âš¡','⚡'
$txt = $txt -replace 'â€”','—'
$txt = $txt -replace 'â€“','–'
$txt = $txt -replace 'â€','"'

# Save clean UTF-8 *without BOM*
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($src, $txt, $utf8NoBom)
Write-Host "[OK] Cleaned encoding."

# Relaunch Streamlit
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
. .\.venv\Scripts\Activate.ps1
python -m streamlit run .\app.py --server.port $Port --server.address localhost
