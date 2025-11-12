# wire-crosses-and-verify.ps1 — ensure SMAs, add cross markers, show counts, align to plot indent
param(
  [string]$Root = "G:\Users\daveq\traderq",
  [string]$App  = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)
$ErrorActionPreference = 'Stop'
Set-Location $Root

# 1) ta_overlays.py with count-returning helper (ASCII only)
@(
'# ta_overlays.py - overlays for Plotly charts (ASCII only)',
'from typing import Tuple',
'import pandas as pd',
'import plotly.graph_objects as go',
'',
'def add_cross_markers_count(fig: go.Figure, df: pd.DataFrame,',
'                            price_col: str = "Close",',
'                            s_short: str = "SMA20",',
'                            s_long: str = "SMA200") -> Tuple[int, int]:',
'    """Add Golden/Death cross markers and return (golden_count, death_count)."""',
'    golden = 0',
'    death = 0',
'    if fig is None or df is None:',
'        return golden, death',
'    for col in (price_col, s_short, s_long):',
'        if col not in df.columns:',
'            return golden, death',
'    prev_short = df[s_short].shift(1)',
'    prev_long  = df[s_long].shift(1)',
'    cross_up   = (prev_short <= prev_long) & (df[s_short] > df[s_long])',
'    cross_down = (prev_short >= prev_long) & (df[s_short] < df[s_long])',
'    x_up = df.index[cross_up];   y_up = df.loc[cross_up, price_col]',
'    x_dn = df.index[cross_down]; y_dn = df.loc[cross_down, price_col]',
'    golden = len(x_up); death = len(x_dn)',
'    if golden > 0:',
'        fig.add_trace(go.Scatter(',
'            x=x_up, y=y_up, mode="markers", name="Golden Cross",',
'            marker=dict(symbol="triangle-up", size=11, color="#17c964", line=dict(width=1, color="#0b3820")),',
'            hovertemplate="Golden cross: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>", showlegend=True))',
'    if death > 0:',
'        fig.add_trace(go.Scatter(',
'            x=x_dn, y=y_dn, mode="markers", name="Death Cross",',
'            marker=dict(symbol="triangle-down", size=11, color="#f31260", line=dict(width=1, color="#4a0b19")),',
'            hovertemplate="Death cross: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>", showlegend=True))',
'    return golden, death',
'',
'# Backward compatible alias',
'def add_cross_markers(fig: go.Figure, df: pd.DataFrame, price_col="Close", s_short="SMA20", s_long="SMA200"):',
'    add_cross_markers_count(fig, df, price_col, s_short, s_long)'
) | Set-Content -Path ".\ta_overlays.py" -Encoding Ascii

# 2) Ensure single import "import ta_overlays as ta" in app.py (after streamlit import if possible)
$appRaw = Get-Content $App -Raw -Encoding UTF8
if ($appRaw -notmatch '^\s*import\s+ta_overlays\s+as\s+ta\b') {
  if ($appRaw -match '^\s*import\s+streamlit\s+as\s+st\s*$') {
    $appRaw = [regex]::Replace($appRaw,'^(?<pre>[ \t]*import\s+streamlit\s+as\s+st\s*\r?\n)','${pre}import ta_overlays as ta  # overlays' + "`r`n",'Multiline')
  } else {
    $appRaw = [regex]::Replace($appRaw,'^(?<head>(?:[ \t]*#.*\r?\n)*[ \t]*from\s+__future__\s+import\s+annotations\s*\r?\n)','${head}import ta_overlays as ta  # overlays' + "`r`n",'Multiline')
  }
}

# 3) Insert pre-plot block: choose price_col, ensure SMAs, call overlay, caption counts
#    We align to the indent of the first st.plotly_chart( line.
$plotMatch = [regex]::Match($appRaw,'^(?<indent>[ \t]*)st\.plotly_chart\s*\(.*$',[System.Text.RegularExpressions.RegexOptions]::Multiline)
if (-not $plotMatch.Success) { throw "Could not find st.plotly_chart( in app.py" }
$pad = $plotMatch.Groups['indent'].Value

$preBlock = ($pad + 'price_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)') + "`r`n" +
            ($pad + 'if price_col is not None:') + "`r`n" +
            ($pad + '    if "SMA20" not in df.columns and price_col in df.columns:') + "`r`n" +
            ($pad + '        df["SMA20"]  = df[price_col].rolling(20, min_periods=20).mean()') + "`r`n" +
            ($pad + '    if "SMA200" not in df.columns and price_col in df.columns:') + "`r`n" +
            ($pad + '        df["SMA200"] = df[price_col].rolling(200, min_periods=200).mean()') + "`r`n" +
            ($pad + '    try:') + "`r`n" +
            ($pad + '        g,d = ta.add_cross_markers_count(fig, df, price_col=price_col, s_short="SMA20", s_long="SMA200")') + "`r`n" +
            ($pad + '        st.caption(f"Crosses: golden={g}, death={d}")') + "`r`n" +
            ($pad + '    except Exception:') + "`r`n" +
            ($pad + '        pass') + "`r`n"

# Remove any previous injected one-liners or old try/except blocks we added earlier
$appRaw = [regex]::Replace($appRaw,'^[ \t]*try:\s*ta\.add_cross_markers\(.*?\)\s*except\s+Exception:\s*pass[ \t]*\r?\n?','', 'Multiline')
$appRaw = [regex]::Replace($appRaw,'^[ \t]*try:\s*\r?\n[ \t]*ta\.add_cross_markers_count\(.*?\)\r?\n[ \t]*except\s+Exception:\s*\r?\n[ \t]*pass\r?\n','', 'Multiline')
$appRaw = [regex]::Replace($appRaw,'^[ \t]*price_col\s*=\s*".*?"\s*if.*\r?\n[ \t]*if\s+price_col\s+is\s+not\s+None:\r?\n(?:.*\r?\n){1,12}?[ \t]*pass\r?\n','', 'Multiline')

# Prepend our block to the plot call line
$appRaw = [regex]::Replace(
  $appRaw,
  '^(?<indent>[ \t]*)st\.plotly_chart\s*\(',
  $preBlock + '${indent}st.plotly_chart(',
  [System.Text.RegularExpressions.RegexOptions]::Multiline
)

# Save back
$appRaw | Set-Content -Path $App -Encoding UTF8

# 4) Restart app so you see it live
Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
