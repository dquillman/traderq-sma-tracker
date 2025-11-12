
# finalize-cross.ps1 — robustly wire SMA cross markers + counts and restart
param(
  [string]$App = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'

# --- 0) Normalize and validate path ---
$App = ($App | Out-String).Trim()
if ([string]::IsNullOrWhiteSpace($App)) { throw "Empty path." }
if ($App -match '[<>:"\|\?\*]') { throw "Illegal characters in path: $App" }
$App = [System.IO.Path]::GetFullPath($App)
if (-not (Test-Path $App)) { throw "Not found: $App" }

# --- 1) Read entire file as raw text ---
$app = Get-Content $App -Raw -Encoding UTF8

# --- 2) Remove any tuple-unpack calls we may have added earlier ---
$app = [regex]::Replace(
  $app,
  '^[ \t]*gu,\s*gd\s*=\s*add_cross_markers\s*\(.*?\)\s*\r?\n',
  '',
  'Multiline'
)

# --- 3) Build safe marker+counts block aligned to the first st.plotly_chart ---
$pm = [regex]::Match($app,'^(?<indent>[ \t]*)st\.plotly_chart\s*\(.*$', 'Multiline')
if (-not $pm.Success) { throw "Could not find st.plotly_chart(" }
$pad  = $pm.Groups['indent'].Value
$pad4 = $pad + '    '

$block =
($pad + '# --- SMA cross markers (safe) ---') + "`r`n" +
($pad + 'try:') + "`r`n" +
($pad4 + '# ensure we know price & SMA columns; assume these exist if you already compute SMAs') + "`r`n" +
($pad4 + 'price_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)') + "`r`n" +
($pad4 + 'if price_col is not None:') + "`r`n" +
($pad4 + '    if "SMA20" not in df.columns:') + "`r`n" +
($pad4 + '        df["SMA20"]  = df[price_col].rolling(20, min_periods=20).mean()') + "`r`n" +
($pad4 + '    if "SMA200" not in df.columns:') + "`r`n" +
($pad4 + '        df["SMA200"] = df[price_col].rolling(200, min_periods=200).mean()') + "`r`n" +
($pad4 + '    add_cross_markers(fig, df, price_col=price_col, s20="SMA20", s200="SMA200")') + "`r`n" +
($pad4 + '    s_prev = df["SMA20"].shift(1); l_prev = df["SMA200"].shift(1)') + "`r`n" +
($pad4 + '    gu = int(((s_prev <= l_prev) & (df["SMA20"] > df["SMA200"])).sum())') + "`r`n" +
($pad4 + '    gd = int(((s_prev >= l_prev) & (df["SMA20"] < df["SMA200"])).sum())') + "`r`n" +
($pad4 + '    st.caption(f"Crosses: golden={gu}, death={gd}  | price={price_col}, s20=SMA20, s200=SMA200")') + "`r`n" +
($pad + 'except Exception:') + "`r`n" +
($pad4 + 'pass') + "`r`n"

# If we already injected a similar block earlier, strip it before re-inserting
$app = [regex]::Replace(
  $app,
  '^[ \t]*# --- SMA cross markers \(safe\) ---[\s\S]*?^[ \t]*pass\r?\n',
  '',
  [System.Text.RegularExpressions.RegexOptions]::Multiline
)

# Prepend our block to the st.plotly_chart line
$app = [regex]::Replace(
  $app,
  '^(?<indent>[ \t]*)st\.plotly_chart\s*\(',
  $block + '${indent}st.plotly_chart(',
  [System.Text.RegularExpressions.RegexOptions]::Multiline
)

# --- 4) Write back using .NET UTF8 (no BOM) to avoid Set-Content quirks ---
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($App, $app, $utf8NoBom)

# --- 5) Restart Streamlit ---
Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = "G:\Users\daveq\traderq\.venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
