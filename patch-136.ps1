# patch-136.ps1 — TraderQ v1.3.6
# - Fixes previous Regex overload error
# - Sets Trend to "Bullish"/"Bearish" (ASCII) and styles cells
# - Rewrites the screener render block with correct indentation
# - Bumps APP_VERSION to v1.3.6
param(
  [switch]$Run,
  [int]$Port = 8578
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

# Load
$txt = Get-Content -Raw -Encoding UTF8 $AppPath

# 1) Bump APP_VERSION safely to v1.3.6
$txt = [System.Text.RegularExpressions.Regex]::new('APP_VERSION\s*=\s*".*?"').Replace($txt,'APP_VERSION = "v1.3.6"',1)

# 2) Normalize Trend creation to Bullish/Bearish (ASCII) — robust single-line match
$trendRegex = [System.Text.RegularExpressions.Regex]::new('"Trend"\s*:\s*.*?,',[System.Text.RegularExpressions.RegexOptions]::Singleline)
$trendExpr  = '"Trend": ("Bullish" if (not d.empty and not np.isnan(d["SMA20"].iloc[-1]) and not np.isnan(d["SMA200"].iloc[-1]) and d["SMA20"].iloc[-1] > d["SMA200"].iloc[-1]) else ("Bearish" if (not d.empty and not np.isnan(d["SMA20"].iloc[-1]) and not np.isnan(d["SMA200"].iloc[-1])) else "-")),'
$txt = $trendRegex.Replace($txt, $trendExpr, 1)

# 3) Replace the Pretouch Screener render block with a styled DataFrame (proper indentation)
$renderPat = [System.Text.RegularExpressions.Regex]::new(
  'st\.subheader\([\s\S]*?Pretouch Screener[\s\S]*?\)\s*(?:\r?\n\s*)?(?:st\.dataframe\([\s\S]*?\)\s*)?(?:\r?\n\s*)?(?:st\.download_button\([\s\S]*?\)\s*)?',
  [System.Text.RegularExpressions.RegexOptions]::Singleline
)
$renderReplacement = @"
st.subheader("Pretouch Screener (closest to SMA200 on top)")

styled = screener_df[["Ticker","Trend","Dist to SMA200 (%)","Dist to SMA20 (%)"]].style.apply(
    lambda s: [
        ("background-color:#12391a;color:#1fd16c;font-weight:600" if v == "Bullish"
         else ("background-color:#3a1919;color:#ff4d4f;font-weight:600" if v == "Bearish" else ""))
        for v in s
    ],
    subset=["Trend"]
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.download_button(
    "Download Screener CSV",
    screener_df.to_csv(index=False).encode("utf-8"),
    file_name="pretouch_screener.csv",
    mime="text/csv",
    key="screener_csv"
)
"@
if ($renderPat.IsMatch($txt)) {
  $txt = $renderPat.Replace($txt, $renderReplacement, 1)
  Write-Host "[OK] Replaced screener render block."
} else {
  Write-Host "[i] Screener subheader not found; no render block replaced."
}

# Save
$txt | Set-Content -Encoding UTF8 $AppPath
Write-Host "[OK] Wrote app.py v1.3.6 (Bullish/Bearish chips + styled screener)."

# Optional: run clean
if ($Run) {
  & taskkill /IM streamlit.exe /F 2>$null | Out-Null
  & taskkill /IM python.exe    /F 2>$null | Out-Null

  $venv = Join-Path $AppDir ".venv\Scripts\Activate.ps1"
  if (!(Test-Path $venv)) {
    & "G:\Python311\python.exe" -m venv (Join-Path $AppDir ".venv")
  }
  & $venv
  python -m pip install --upgrade pip
  if (Test-Path (Join-Path $AppDir "requirements.txt")) {
    python -m pip install --no-cache-dir -r (Join-Path $AppDir "requirements.txt")
  }
  streamlit cache clear
  python -m streamlit run $AppPath --server.port $Port
}
