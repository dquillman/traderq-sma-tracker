# fix-config-hard.ps1 — remove bad .streamlit/config.toml, rewrite clean ASCII, relaunch
param([int]$Port = 8617)

$ErrorActionPreference = "Stop"
Set-Location G:\Users\daveq\traderq

# 1) Kill any running streamlit
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# 2) Ensure .streamlit exists
$stDir = ".\.streamlit"
if (!(Test-Path $stDir)) { New-Item -ItemType Directory -Path $stDir | Out-Null }

# 3) Delete any existing config.toml (nuke the problem)
$configPath = Join-Path $stDir "config.toml"
Remove-Item -LiteralPath $configPath -Force -ErrorAction SilentlyContinue

# 4) Write a CLEAN config.toml using ASCII (safe, no BOM)
$clean = @"
[theme]
base = "dark"
primaryColor = "#6ae3ff"
backgroundColor = "#0b1020"
secondaryBackgroundColor = "#11162a"
textColor = "#e8ecf3"
"@

$bytes = [System.Text.Encoding]::ASCII.GetBytes($clean)
[System.IO.File]::WriteAllBytes($configPath, $bytes)

# 5) Verify first byte is '[' (0x5B). If not, fail fast.
$first = [System.IO.File]::ReadAllBytes($configPath)[0]
if ($first -ne 0x5B) {
  throw "config.toml still has a bad first byte ($first)."
}

# 6) Activate venv & relaunch
. .\.venv\Scripts\Activate.ps1
python -m streamlit run .\app.py --server.port $Port --server.address localhost
