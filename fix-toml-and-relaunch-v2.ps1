# fix-toml-and-relaunch-v2.ps1 — write .streamlit/config.toml w/o BOM and relaunch
param([int]$Port = 8617)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

# Ensure folder
$stDir = ".\.streamlit"
if (!(Test-Path $stDir)) { New-Item -ItemType Directory -Path $stDir | Out-Null }

# Valid TOML content
$config = @"
[theme]
base = "dark"
primaryColor = "#6ae3ff"
backgroundColor = "#0b1020"
secondaryBackgroundColor = "#11162a"
textColor = "#e8ecf3"
"@

# Write WITHOUT BOM using .NET
$configPath = Join-Path $stDir "config.toml"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($configPath, $config, $utf8NoBom)

# Kill any running Streamlit and relaunch
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
. .\.venv\Scripts\Activate.ps1
python -m streamlit run .\app.py --server.port $Port --server.address localhost
