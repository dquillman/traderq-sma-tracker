# fix-header-html.ps1 — wraps stray header HTML in a proper st.markdown block and relaunches
param([int]$Port = 8617)

$ErrorActionPreference = "Stop"
Set-Location G:\Users\daveq\traderq
$src = "app.py"
if (!(Test-Path $src)) { throw "app.py not found" }

# 1) Backup
$bak = "$src.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $src $bak -Force
Write-Host "[OK] Backup -> $bak"

# 2) Load
$txt = Get-Content $src -Raw -Encoding UTF8

# 3) Ensure APP_VERSION exists (minimal guard)
if ($txt -notmatch 'APP_VERSION\s*=\s*"') {
  $txt = 'APP_VERSION = "v1.7.1"' + "`r`n" + $txt
}

# 4) Build the correct Python block for the header
$pyHeader = @"
st.markdown(
    f'''
    <div class="traderq-nav">
      <div class="traderq-brand">⚡ TraderQ — SMA 20/200</div>
      <div class="traderq-pill">Version: <b>{APP_VERSION}</b></div>
    </div>
    ''',
    unsafe_allow_html=True
)
"@

# 5) Replace any raw header HTML block with the python block
#    Matches lines starting with <div class="traderq-nav"> up to the closing </div> of the container
$pattern = '(?ms)^\s*<div class="traderq-nav">.*?</div>\s*$'
$replaced = [System.Text.RegularExpressions.Regex]::Replace($txt, $pattern, $pyHeader)

# 6) If nothing changed (header not found raw), insert just after set_page_config or at top
if ($replaced -eq $txt) {
  $cfg = 'st.set_page_config('
  $idx = $txt.IndexOf($cfg)
  if ($idx -ge 0) {
    $lineEnd = $txt.IndexOf("`n", $idx); if ($lineEnd -lt 0){ $lineEnd = $idx + $cfg.Length }
    $replaced = $txt.Insert($lineEnd+1, "`r`n$pyHeader`r`n")
  } else {
    $replaced = "$pyHeader`r`n$txt"
  }
}

# 7) Save
Set-Content $src $replaced -Encoding UTF8
Write-Host "[OK] Header HTML wrapped correctly."

# 8) Relaunch Streamlit
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
. .\.venv\Scripts\Activate.ps1
python -m streamlit run .\app.py --server.port $Port --server.address localhost
