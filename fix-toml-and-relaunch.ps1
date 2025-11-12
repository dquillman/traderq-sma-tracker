# fix-toml-and-relaunch.ps1 — rewrite .streamlit/config.toml without BOM and relaunch
param(
  [int]$Port = 8617
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

# Ensure folder
$stDir = ".\.streamlit"
if (!(Test-Path $stDir)) { New-Item -ItemType Directory -Path $stDir | Out-Null }

# Valid TOML (no weird chars)
$config = @"
[theme]
base = "dark"
primaryColor = "#6ae3ff"
backgroundColor = "#0b1020"
secondaryBackgroundColor = "#11162a"
textColor = "#e8ecf3"
"@

# Write **without BOM**
$configPath = Join-Path $stDir "config.toml"
Set-Content -LiteralPath $configPath -Value $config -Encoding utf8NoBOM

# Relaunch Streamlit
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
. .\.venv\Scripts\Activate.ps1
python -m streamlit run .\app.py --server.port $Port --server.address localhost

