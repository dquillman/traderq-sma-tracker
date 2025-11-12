# patch-134.ps1 — Fix Trend column mojibake (use ASCII arrows + styled cells), bump v1.3.4
param(
  [switch]$Run,
  [int]$Port = 8576
)

$ErrorActionPreference = 'Stop'
$AppDir  = "G:\Users\daveq\traderq"
$AppPath = Join-Path $AppDir "app.py"
if (!(Test-Path $AppPath)) { Write-Error "app.py not found at $AppPath"; exit 1 }

# Backup
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$AppPath.bak_$ts"
Copy-Item $AppPath $bak
Write-Host "[OK] Backup -> $bak"

# Load file
$txt = Get-Content -Raw -Encoding UTF8 $AppPath

# 1) Bump APP_VERSION to v1.3.4
$txt = [regex]::Replace($txt,'APP_VERSION\s*=\s*".*?"','APP_VERSION = "v1.3.4"',1)

# 2) Replace emoji in logic with ASCII arrows (▲ / ▼) to avoid mojibake
#    We’ll do a broad replace in case those strings appear in two places.
$txt = $txt -replace '🟢','▲'
$txt = $txt -replace '🔴','▼'

# 3) Ensure the Screener builds Trend using ▲/▼ (catch common variants)
$txt = [regex]::Replace(
  $txt,
  'trend\s*=\s*"(?:🟢|🔴)".*?else\s*"(?:🟢|🔴)"',
  'trend = "▲" if d["SMA20"].iloc[-1] > d["SMA200"].iloc[-1] else "▼"',
  [System.Text.RegularExpressions.RegexOptions]::Singleline
)

# 4) Replace st.dataframe(screener_df[...]) with a styled version that colors Trend cells
#    Find the most specific call that renders the screener subset and replace it.
$pattern = 'st\.dataframe\(\s*screener_df\[\s*\[\s*"Ticker"\s*,\s*"Trend"\s*,\s*"Dist to SMA200 \(%\)"\s*,\s*"Dist to SMA20 \(%\)"\s*\]\s*\]\s*,\s*use_container_width=True\s*,\s*hide_index=True\s*\)'
$replacement = @"
styled = screener_df[["Ticker","Trend","Dist to SMA200 (%)","Dist to SMA20 (%)"]].style.apply(
    lambda s: ["background-color:#12391a;color:#1fd16c;font-weight:600" if v=="▲" else ("background-color:#3a1919;color:#ff4d4f;font-weight:600" if v=="▼" else "") for v in s],
    subset=["Trend"]
)
st.dataframe(styled, use_container_width=True, hide_index=True)
"@
if ($txt -match $pattern) {
  $txt = [regex]::Replace($txt, $pattern, $replacement, 1)
} else {
  # Fallback: try a looser match (in case your subset differs)
  $pattern2 = 'st\.dataframe\(\s*screener_df[^\)]*\)'
  if ($txt -match $pattern2) {
    $txt = [regex]::Replace($txt, $pattern2, $replacement, 1)
  } else {
    Write-Host "[i] Could not find screener st.dataframe(...) to replace. Leaving as-is."
  }
}

# Save file
$txt | Set-Content -Encoding UTF8 $AppPath
Write-Host "[OK] Wrote app.py v1.3.4 with styled Trend column (▲/▼)."

# Optional run
if ($Run) {
  # Kill any old servers
  & taskkill /IM streamlit.exe /F 2>$null | Out-Null
  & taskkill /IM python.exe    /F 2>$null | Out-Null

  # Activate venv & run
  $venv = Join-Path $AppDir ".venv\Scripts\Activate.ps1"
  if (!(Test-Path $venv)) {
    & "G:\Python311\python.exe" -m venv (Join-Path $AppDir ".venv")
  }
  & $venv
  python -m pip install --upgrade pip
  # Requirements already set by your previous patch; no changes here
  streamlit cache clear
  python -m streamlit run $AppPath --server.port $Port
}
