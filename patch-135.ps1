# patch-135.ps1 — TraderQ v1.3.5: fix screener indentation + switch Trend to text chips with styling
param(
  [switch]$Run,
  [int]$Port = 8577
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

# 1) Bump version to v1.3.5
$txt = [regex]::Replace($txt,'APP_VERSION\s*=\s*".*?"','APP_VERSION = "v1.3.5"',1)

# 2) Force Trend to Bullish/Bearish text (no emoji) in screener row builder
#    Replace any existing Trend expression with a robust one
$trendExpr = '"Trend": ("Bullish" if (not d.empty and not np.isnan(d["SMA20"].iloc[-1]) and not np.isnan(d["SMA200"].iloc[-1]) and d["SMA20"].iloc[-1] > d["SMA200"].iloc[-1]) else ("Bearish" if (not d.empty and not np.isnan(d["SMA20"].iloc[-1]) and not np.isnan(d["SMA200"].iloc[-1])) else "-")),'
$txt = [regex]::Replace($txt, '"Trend"\s*:\s*.*?,', $trendExpr, 1, [System.Text.RegularExpressions.RegexOptions]::Singleline)

# 3) Replace the screener display block with a styled DataFrame (proper indentation)
#    Find the block from st.subheader("...Pretouch Screener...") through the following st.dataframe(...) and st.download_button(...).
$pattern = 'st\.subheader\([\s\S]*?Pretouch Screener[\s\S]*?\)\s*' +       # the subheader line
           '(?:\r?\n\s*)?' +
           '(?:st\.dataframe\([\s\S]*?hide_index=True\)\s*)?' +            # optional old dataframe
           '(?:\r?\n\s*)?' +
           '(?:st\.download_button\([\s\S]*?\)\s*)?'                        # optional old download button
$replacement = @'
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
'@

if ($txt -match $pattern) {
  # Replace the FIRST matching block after the screener_df is created.
  # Safer approach: perform replacement only after the line that builds screener_df.
  $beforeAfter = $txt -split '(screener_df\s*=\s*pd\.DataFrame\([^\)]*\)[\s\S]*?sort_values\([^\)]*\)\s*)', 2
  if ($beforeAfter.Count -eq 3) {
    $head = $beforeAfter[0] + $beforeAfter[1]
    $tail = $beforeAfter[2]
    $tail = [regex]::Replace($tail, $pattern, $replacement, 1)
    $txt  = $head + $tail
    Write-Host "[OK] Rewrote screener render block with styled table."
  } else {
    # Fallback: global (single) replacement
    $txt = [regex]::Replace($txt, $pattern, $replacement, 1)
    Write-Host "[i] Could not isolate screener_df block precisely; replaced first Pretouch Screener render globally."
  }
} else {
  Write-Host "[i] Pretouch Screener subheader not found; no screener render replaced."
}

# Save
$txt | Set-Content -Encoding UTF8 $AppPath
Write-Host "[OK] Wrote app.py v1.3.5 (styled Trend chips, fixed indentation)."

# Optional run
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
