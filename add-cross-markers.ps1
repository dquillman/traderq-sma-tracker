# add-cross-markers.ps1 — safely add SMA cross markers to your chart and restart
param(
  [string]$App = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'
if (!(Test-Path $App)) { throw "Not found: $App" }

# Read full file
$appRaw = Get-Content $App -Raw -Encoding UTF8

# 1) Insert helper def add_cross_markers(...) once (right after 'import plotly.graph_objects as go')
if ($appRaw -notmatch 'def\s+add_cross_markers\s*\(') {
  $m = [regex]::Match($appRaw, '^[ \t]*import\s+plotly\.graph_objects\s+as\s+go\s*\r?$', 'Multiline')
  if (-not $m.Success) { throw "Could not find 'import plotly.graph_objects as go' to anchor helper insertion." }

  $helper = @(
    'def add_cross_markers(fig: go.Figure, df: pd.DataFrame,',
    '                      price_col: str = "Close",',
    '                      s20: str = "SMA20",',
    '                      s200: str = "SMA200") -> None:',
    '    """',
    '    Golden cross: 20 crosses above 200 (green triangle-up)',
    '    Death  cross: 20 crosses below 200 (red triangle-down)',
    '    Works if df has df[price_col], df[s20], df[s200].',
    '    """',
    '    if df is None or df.empty:',
    '        return',
    '    needed = (price_col, s20, s200)',
    '    if any(c not in df.columns for c in needed):',
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
  $appRaw = $appRaw.Substring(0, $insertPos) + "`r`n`r`n" + $helper + "`r`n`r`n" + $appRaw.Substring($insertPos)
}

# 2) Insert call block before the FIRST st.plotly_chart(, only if not present already
if ($appRaw -notmatch 'add_cross_markers\s*\(') {
  $pm = [regex]::Match($appRaw, '^(?<indent>[ \t]*)st\.plotly_chart\s*\(.*$', 'Multiline')
  if (-not $pm.Success) { throw "Could not find st.plotly_chart( to anchor marker call." }
  $pad = $pm.Groups['indent'].Value

  $preBlock = ($pad + 'price_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)') + "`r`n" +
              ($pad + 'if price_col is not None:') + "`r`n" +
              ($pad + '    if "SMA20" not in df.columns:') + "`r`n" +
              ($pad + '        df["SMA20"] = df[price_col].rolling(20, min_periods=20).mean()') + "`r`n" +
              ($pad + '    if "SMA200" not in df.columns:') + "`r`n" +
              ($pad + '        df["SMA200"] = df[price_col].rolling(200, min_periods=200).mean()') + "`r`n" +
              ($pad + '    add_cross_markers(fig, df, price_col=price_col)') + "`r`n"

  # Prepend our block to that line
  $appRaw = [regex]::Replace(
    $appRaw,
    '^(?<indent>[ \t]*)st\.plotly_chart\s*\(',
    $preBlock + '${indent}st.plotly_chart(',
    [System.Text.RegularExpressions.RegexOptions]::Multiline
  )
}

# 3) Write back and restart
$appRaw | Set-Content -Encoding UTF8 $App

Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = "G:\Users\daveq\traderq\.venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
