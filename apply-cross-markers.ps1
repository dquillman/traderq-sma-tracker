# apply-cross-markers.ps1  — add Golden/Death cross markers to the Plotly chart
param(
  [string]$Root = "G:\Users\daveq\traderq",
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'
Set-Location $Root

# 1) Create ta_overlays.py (ASCII only)
$ta = @(
'# ta_overlays.py — overlays for Plotly charts (ASCII only)',
'from typing import Optional',
'import pandas as pd',
'import plotly.graph_objects as go',
'',
'def add_cross_markers(fig: go.Figure, df: pd.DataFrame,',
'                      price_col: str = "Close",',
'                      s_short: str = "SMA20",',
'                      s_long: str = "SMA200") -> None:',
'    """',
'    Adds Golden/Death cross markers based on short vs long SMAs.',
'      - Golden cross: short > long  (triangle-up, green)',
'      - Death  cross: short < long  (triangle-down, red)',
'    Requires df with columns: price_col, s_short, s_long.',
'    """',
'    if fig is None or df is None:',
'        return',
'    for col in (price_col, s_short, s_long):',
'        if col not in df.columns:',
'            return',
'    prev_short = df[s_short].shift(1)',
'    prev_long  = df[s_long].shift(1)',
'    cross_up   = (prev_short <= prev_long) & (df[s_short] > df[s_long])',
'    cross_down = (prev_short >= prev_long) & (df[s_short] < df[s_long])',
'',
'    x_up = df.index[cross_up]',
'    y_up = df.loc[cross_up, price_col]',
'    x_dn = df.index[cross_down]',
'    y_dn = df.loc[cross_down, price_col]',
'',
'    if len(x_up) > 0:',
'        fig.add_trace(go.Scatter(',
'            x=x_up, y=y_up, mode="markers", name="Golden Cross",',
'            marker=dict(symbol="triangle-up", size=11, color="#17c964", line=dict(width=1, color="#0b3820")),',
'            hovertemplate="Golden cross: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",',
'            showlegend=True',
'        ))',
'    if len(x_dn) > 0:',
'        fig.add_trace(go.Scatter(',
'            x=x_dn, y=y_dn, mode="markers", name="Death Cross",',
'            marker=dict(symbol="triangle-down", size=11, color="#f31260", line=dict(width=1, color="#4a0b19")),',
'            hovertemplate="Death cross: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",',
'            showlegend=True',
'        ))',
''
)
$ta | Set-Content -Path ".\ta_overlays.py" -Encoding Ascii

# 2) Patch app.py: import and call
$app = ".\app.py"
if(!(Test-Path $app)){ throw "app.py not found at $Root" }
$lines = Get-Content $app -Encoding UTF8

# Ensure single import 'import ta_overlays as ta' after 'import streamlit as st'
$lines = $lines | Where-Object { $_ -notmatch '^\s*import\s+ta_overlays\b' }
$stLine = ($lines | Select-String -Pattern '^\s*import\s+streamlit\s+as\s+st\s*$').LineNumber
if($stLine){
  $i = [int]$stLine[0]-1
  $pre  = $lines[0..$i]
  $post = if($i+1 -lt $lines.Count){ $lines[($i+1)..($lines.Count-1)] } else { @() }
  $lines = @(); $lines += $pre; $lines += 'import ta_overlays as ta  # cross markers'; $lines += $post
} else {
  $lines = @('import ta_overlays as ta  # cross markers') + $lines
}

# Insert marker call BEFORE first st.plotly_chart(...), wrapped in try/except.
# We assume main DataFrame is named "df" and has Close/SMA20/SMA200; safe-guard if not.
$joined = $lines -join "`n"
if($joined -notmatch 'ta\.add_cross_markers\('){
  $out = @()
  $inserted = $false
  foreach($ln in $lines){
    if(-not $inserted -and $ln -match 'st\.plotly_chart\s*\('){
      $out += 'try:'
      $out += '    ta.add_cross_markers(fig, df, price_col="Close", s_short="SMA20", s_long="SMA200")'
      $out += 'except Exception:'
      $out += '    pass'
      $inserted = $true
    }
    $out += $ln
  }
  if(-not $inserted){
    # Fallback: append at end (still safe)
    $out += ''
    $out += '# cross markers fallback (no-op if fig/df not in scope)'
    $out += 'try:'
    $out += '    ta.add_cross_markers(fig, df, price_col="Close", s_short="SMA20", s_long="SMA200")'
    $out += 'except Exception:'
    $out += '    pass'
  }
  $lines = $out
}

$lines | Set-Content -Path $app -Encoding UTF8

# 3) Restart app
Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m streamlit run .\app.py --server.port $Port
