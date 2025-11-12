# add-cross-markers-final.ps1 — add SMA20/200 cross markers + counts, restart app
param(
  [string]$App = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'

# Normalize path and read file
$App = ($App | Out-String).Trim()
if (-not (Test-Path $App)) { throw "Not found: $App" }
$App = [System.IO.Path]::GetFullPath($App)
$text = Get-Content $App -Raw -Encoding UTF8

# 1) Insert helper once (anchor: 'import plotly.graph_objects as go')
if ($text -notmatch 'def\s+add_cross_markers\s*\(') {
  $m = [regex]::Match($text,'^[ \t]*import\s+plotly\.graph_objects\s+as\s+go\s*\r?$', 'Multiline')
  if (-not $m.Success) { throw "Anchor not found: import plotly.graph_objects as go" }

  $helper = @(
'def add_cross_markers(fig: go.Figure, df: pd.DataFrame,',
'                      price_col: str = "Close",',
'                      s20: str = "SMA20",',
'                      s200: str = "SMA200") -> None:',
'    """',
'    Golden cross: 20 crosses above 200 (green triangle-up)',
'    Death  cross: 20 crosses below 200 (red triangle-down)',
'    Requires df[price_col], df[s20], df[s200].',
'    """',
'    if df is None or df.empty:',
'        return',
'    if any(c not in df.columns for c in (price_col, s20, s200)):',
'        return',
'    s_prev = df[s20].shift(1)',
'    l_prev = df[s200].shift(1)',
'    cross_up = (s_prev <= l_prev) & (df[s20] > df[s200])',
'    cross_dn = (s_prev >= l_prev) & (df[s20] < df[s200])',
'    xu, yu = df.index[cross_up], df.loc[cross_up, price_col]',
'    xd, yd = df.index[cross_dn], df.loc[cross_dn, price_col]',
'    if len(xu):',
'        fig.add_trace(go.Scatter(',
'            x=xu, y=yu, mode="markers", name="Golden Cross",',
'            marker=dict(symbol="triangle-up", size=11, color="#17c964",',
'                        line=dict(width=1, color="#0b3820")),',
'            hovertemplate="Golden: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",',
'            showlegend=True',
'        ))',
'    if len(xd):',
'        fig.add_trace(go.Scatter(',
'            x=xd, y=yd, mode="markers", name="Death Cross",',
'            marker=dict(symbol="triangle-down", size=11, color="#f31260",',
'                        line=dict(width=1, color="#4a0b19")),',
'            hovertemplate="Death: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",',
'            showlegend=True',
'        ))'
  ) -join "`r`n"

  $insertPos = $m.Index + $m.Length
  $text = $text.Substring(0,$insertPos) + "`r`n`r`n" + $helper + "`r`n`r`n" + $text.Substring($insertPos)
}

# 2) Remove any previous broken injections we might have added earlier
$text = [regex]::Replace($text,'^[ \t]*gu,\s*gd\s*=\s*add_cross_markers\s*\(.*?\)\s*\r?\n','', 'Multiline')
$text = [regex]::Replace($text,'^[ \t]*# --- SMA (?:20/200 )?cross markers.*?^[ \t]*pass\r?\n','', [System.Text.RegularExpressions.RegexOptions]::Multiline)

# 3) Build safe block aligned to the first st.plotly_chart(
$pm = [regex]::Match($text,'^(?<indent>[ \t]*)st\.plotly_chart\s*\(.*$', 'Multiline')
if (-not $pm.Success) { throw "Could not find st.plotly_chart(" }
$pad  = $pm.Groups['indent'].Value
$pad4 = $pad + '    '

$preBlock =
($pad + '# --- SMA20/200 markers + counts (safe) ---') + "`r`n" +
($pad + 'price_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)') + "`r`n" +
($pad + 'if price_col is not None:') + "`r`n" +
($pad4 + 'if "SMA20" not in df.columns:') + "`r`n" +
($pad4 + '    df["SMA20"]  = df[price_col].rolling(20, min_periods=20).mean()') + "`r`n" +
($pad4 + 'if "SMA200" not in df.columns:') + "`r`n" +
($pad4 + '    df["SMA200"] = df[price_col].rolling(200, min_periods=200).mean()') + "`r`n" +
($pad4 + 'add_cross_markers(fig, df, price_col=price_col, s20="SMA20", s200="SMA200")') + "`r`n" +
($pad4 + '_s_prev = df["SMA20"].shift(1); _l_prev = df["SMA200"].shift(1)') + "`r`n" +
($pad4 + '_golden = int(((_s_prev <= _l_prev) & (df["SMA20"] > df["SMA200"])).sum())') + "`r`n" +
($pad4 + '_death  = int(((_s_prev >= _l_prev) & (df["SMA20"] < df["SMA200"])).sum())') + "`r`n" +
($pad4 + 'st.caption(f"Crosses: golden={_golden}, death={_death} | price={price_col}, s20=SMA20, s200=SMA200")') + "`r`n"

# Inject block before the st.plotly_chart line
$text = [regex]::Replace(
  $text,
  '^(?<indent>[ \t]*)st\.plotly_chart\s*\(',
  $preBlock + '${indent}st.plotly_chart(',
  [System.Text.RegularExpressions.RegexOptions]::Multiline
)

# 4) Write back (UTF-8 no BOM), restart Streamlit
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($App, $text, $utf8NoBom)

Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = (Join-Path ([System.IO.Path]::GetDirectoryName($App)) ".venv\Scripts\python.exe")
& $venvPy -m streamlit run $App --server.port $Port
