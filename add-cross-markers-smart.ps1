# add-cross-markers-smart.ps1 — auto-detect price/SMA columns, add markers, show counts, restart
param(
  [string]$App = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)
$ErrorActionPreference = 'Stop'
if(!(Test-Path $App)){ throw "Not found: $App" }

# Read file
$appRaw = Get-Content $App -Raw -Encoding UTF8

# Ensure helper exists (once) right after "import plotly.graph_objects as go"
if($appRaw -notmatch 'def\s+add_cross_markers\s*\('){
  $m = [regex]::Match($appRaw,'^[ \t]*import\s+plotly\.graph_objects\s+as\s+go\s*\r?$', 'Multiline')
  if(-not $m.Success){ throw "Anchor not found: import plotly.graph_objects as go" }
  $helper = @(
    'def add_cross_markers(fig: go.Figure, df: pd.DataFrame, price_col: str, s20: str, s200: str):',
    '    if df is None or df.empty: return 0,0',
    '    for c in (price_col, s20, s200):',
    '        if c not in df.columns: return 0,0',
    '    s_prev = df[s20].shift(1); l_prev = df[s200].shift(1)',
    '    cross_up = (s_prev <= l_prev) & (df[s20] > df[s200])',
    '    cross_dn = (s_prev >= l_prev) & (df[s20] < df[s200])',
    '    xu, yu = df.index[cross_up], df.loc[cross_up, price_col]',
    '    xd, yd = df.index[cross_dn], df.loc[cross_dn, price_col]',
    '    gu = len(xu); gd = len(xd)',
    '    if gu:',
    '        fig.add_trace(go.Scatter(x=xu, y=yu, mode="markers", name="Golden Cross",',
    '            marker=dict(symbol="triangle-up", size=11, color="#17c964", line=dict(width=1, color="#0b3820")),',
    '            hovertemplate="Golden: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>", showlegend=True))',
    '    if gd:',
    '        fig.add_trace(go.Scatter(x=xd, y=yd, mode="markers", name="Death Cross",',
    '            marker=dict(symbol="triangle-down", size=11, color="#f31260", line=dict(width=1, color="#4a0b19")),',
    '            hovertemplate="Death: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>", showlegend=True))',
    '    return gu, gd'
  ) -join "`r`n"
  $insertPos = $m.Index + $m.Length
  $appRaw = $appRaw.Substring(0,$insertPos) + "`r`n`r`n" + $helper + "`r`n`r`n" + $appRaw.Substring($insertPos)
}

# Build robust pre-plot block (auto-detect columns, ensure SMAs, add markers, caption counts)
$pm = [regex]::Match($appRaw,'^(?<indent>[ \t]*)st\.plotly_chart\s*\(.*$', 'Multiline')
if(-not $pm.Success){ throw "Anchor not found: st.plotly_chart(" }
$pad = $pm.Groups['indent'].Value

$preBlock = ($pad + '# --- auto SMA cross markers ---') + "`r`n" +
($pad + 'cols_lower = {c.lower(): c for c in df.columns}') + "`r`n" +
($pad + 'price_candidates = ["close","adj close","adjclose","adj_close","price","c"]') + "`r`n" +
($pad + 'price_col = None') + "`r`n" +
($pad + 'for cand in price_candidates:') + "`r`n" +
($pad + '    if cand in cols_lower:') + "`r`n" +
($pad + '        price_col = cols_lower[cand]; break') + "`r`n" +
($pad + 'if price_col is not None:') + "`r`n" +
($pad + '    # find SMA20/SMA200 by name if present (case-insensitive), else create') + "`r`n" +
($pad + '    s20 = None; s200 = None') + "`r`n" +
($pad + '    for c in df.columns:') + "`r`n" +
($pad + '        lc = c.lower()') + "`r`n" +
($pad + '        if "sma" in lc and "20" in lc: s20 = c') + "`r`n" +
($pad + '        if "sma" in lc and "200" in lc: s200 = c') + "`r`n" +
($pad + '    if s20 is None:') + "`r`n" +
($pad + '        s20 = "SMA20"; df[s20] = df[price_col].rolling(20, min_periods=20).mean()') + "`r`n" +
($pad + '    if s200 is None:') + "`r`n" +
($pad + '        s200 = "SMA200"; df[s200] = df[price_col].rolling(200, min_periods=200).mean()') + "`r`n" +
($pad + '    gu, gd = add_cross_markers(fig, df, price_col, s20, s200)') + "`r`n" +
($pad + '    st.caption(f"Crosses: golden={gu}, death={gd}  | price={price_col}, s20={s20}, s200={s200}")') + "`r`n"

# Remove any prior injected one-liners or blocks we might have added
$appRaw = [regex]::Replace($appRaw,'^[ \t]*try:\s*ta\.add_cross_markers\(.*?\)\s*except\s+Exception:\s*pass[ \t]*\r?\n?','', 'Multiline')
$appRaw = [regex]::Replace($appRaw,'^[ \t]*price_col\s*=\s*".*?"\s*if.*\r?\n[ \t]*if\s+price_col\s+is\s+not\s+None:.*?add_cross_markers\(.*?\)\r?\n','', 'Singleline')

# Inject block before st.plotly_chart
$appRaw = [regex]::Replace(
  $appRaw,
  '^(?<indent>[ \t]*)st\.plotly_chart\s*\(',
  $preBlock + '${indent}st.plotly_chart(',
  [System.Text.RegularExpressions.RegexOptions]::Multiline
)

# Save & restart
$appRaw | Set-Content -Encoding UTF8 $App
Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = "G:\Users\daveq\traderq\.venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
