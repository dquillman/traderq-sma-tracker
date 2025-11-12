# fix-page-config.ps1 — ensure single, first st.set_page_config; bump v1.6.1; optional run
param(
  [switch]$Run,
  [int]$Port = 8616
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq
$src = "app.py"
if (!(Test-Path $src)) { Write-Error "app.py not found"; exit 1 }

# Backup
$bak = "$src.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $src $bak -Force
Write-Host "[OK] Backup -> $bak"

# Load
$text = Get-Content $src -Raw -Encoding UTF8

# 1) Bump version
$text = [regex]::Replace($text,'APP_VERSION\s*=\s*"(?:[^"]*)"', 'APP_VERSION = "v1.6.1"', 1)

# 2) Strip ALL set_page_config calls
$text = [regex]::Replace($text, '^\s*st\.set_page_config\([^\)]*\)\s*$', '', 'Multiline')

# 3) Insert a single canonical set_page_config right after "import streamlit as st"
$needle = "import streamlit as st"
$pos = $text.IndexOf($needle)
if ($pos -lt 0) {
  # if import missing, prepend it
  $text = "$needle`r`n" + $text
  $pos = 0
}
$lineEnd = $text.IndexOf("`n", $pos)
if ($lineEnd -lt 0) { $lineEnd = $pos + $needle.Length }

$cfg = @"
st.set_page_config(page_title="TraderQ SMA 20/200", page_icon="📈", layout="wide")
"@

# Ensure no other Streamlit calls before config
$head = $text.Substring(0, $lineEnd+1)
$tail = $text.Substring($lineEnd+1)
# Remove accidental leading blank lines in tail
$tail = $tail.TrimStart("`r","`n")
$text = $head + $cfg + "`r`n" + $tail

# Save
Set-Content $src $text -Encoding UTF8
Write-Host "[OK] set_page_config normalized and version bumped to v1.6.1"

# Optional: restart
if ($Run) {
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  .\.venv\Scripts\Activate.ps1 | Out-Null
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}
