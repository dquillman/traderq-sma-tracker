# fix-cross-call.ps1 — replace bad "gu, gd = add_cross_markers(...)" with safe block that computes counts inline
param(
  [string]$App = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'
if(!(Test-Path $App)){ throw "Not found: $App" }

# Read whole file
$app = Get-Content $App -Raw -Encoding UTF8

# Find the first st.plotly_chart( to anchor indentation
$pm = [regex]::Match($app,'^(?<indent>[ \t]*)st\.plotly_chart\s*\(.*$', 'Multiline')
if(-not $pm.Success){ throw "Could not find st.plotly_chart(" }
$pad = $pm.Groups['indent'].Value
$pad4 = $pad + '    '

# Remove any line that does tuple-unpack from add_cross_markers(...)
$app = [regex]::Replace(
  $app,
  '^[ \t]*gu,\s*gd\s*=\s*add_cross_markers\s*\(.*?\)\s*\r?\n',
  '',
  'Multiline'
)

# Insert safe block right before st.plotly_chart(
$block =
($pad + '# --- add markers and compute counts inline ---') + "`r`n" +
($pad + 'try:') + "`r`n" +
($pad4 + 'add_cross_markers(fig, df, price_col, s20, s200)') + "`r`n" +
($pad4 + 's_prev = df[s20].shift(1); l_prev = df[s200].shift(1)') + "`r`n" +
($pad4 + 'gu = int(((s_prev <= l_prev) & (df[s20] > df[s200])).sum())') + "`r`n" +
($pad4 + 'gd = int(((s_prev >= l_prev) & (df[s20] < df[s200])).sum())') + "`r`n" +
($pad4 + 'st.caption(f"Crosses: golden={gu}, death={gd}  | price={price_col}, s20={s20}, s200={s200}")') + "`r`n" +
($pad + 'except Exception:') + "`r`n" +
($pad4 + 'pass') + "`r`n"

$app = [regex]::Replace(
  $app,
  '^(?<indent>[ \t]*)st\.plotly_chart\s*\(',
  $block + '${indent}st.plotly_chart(',
  [System.Text.RegularExpressions.RegexOptions]::Multiline
)

# Write back and restart
[System.IO.File]::WriteAllText($App, $app, [System.Text.Encoding]::UTF8)

Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = "G:\Users\daveq\traderq\.venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
