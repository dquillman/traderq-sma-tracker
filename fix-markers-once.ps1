# fix-markers-once.ps1 — remove bad one-liner, insert proper multi-line markers aligned to plot
param(
  [string]$Root = "G:\Users\daveq\traderq",
  [string]$App  = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)
$ErrorActionPreference = 'Stop'
Set-Location $Root

# 0) Ensure overlay helper exists (ASCII)
if(!(Test-Path ".\ta_overlays.py")){
  @(
'# ta_overlays.py — overlays for Plotly charts (ASCII only)',
'from typing import Optional',
'import pandas as pd',
'import plotly.graph_objects as go',
'',
'def add_cross_markers(fig: go.Figure, df: pd.DataFrame,',
'                      price_col: str = "Close",',
'                      s_short: str = "SMA20",',
'                      s_long: str = "SMA200") -> None:',
'    if fig is None or df is None:',
'        return',
'    for col in (price_col, s_short, s_long):',
'        if col not in df.columns:',
'            return',
'    prev_short = df[s_short].shift(1)',
'    prev_long  = df[s_long].shift(1)',
'    cross_up   = (prev_short <= prev_long) & (df[s_short] > df[s_long])',
'    cross_down = (prev_short >= prev_long) & (df[s_short] < df[s_long])',
'    x_up = df.index[cross_up]',
'    y_up = df.loc[cross_up, price_col]',
'    x_dn = df.index[cross_down]',
'    y_dn = df.loc[cross_down, price_col]',
'    if len(x_up) > 0:',
'        fig.add_trace(go.Scatter(x=x_up, y=y_up, mode="markers", name="Golden Cross",',
'            marker=dict(symbol="triangle-up", size=11, color="#17c964", line=dict(width=1, color="#0b3820")),',
'            hovertemplate="Golden cross: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>", showlegend=True))',
'    if len(x_dn) > 0:',
'        fig.add_trace(go.Scatter(x=x_dn, y=y_dn, mode="markers", name="Death Cross",',
'            marker=dict(symbol="triangle-down", size=11, color="#f31260", line=dict(width=1, color="#4a0b19")),',
'            hovertemplate="Death cross: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>", showlegend=True))'
  ) | Set-Content -Encoding Ascii ".\ta_overlays.py"
}

# 1) Load app.py as raw text
if(!(Test-Path $App)){ throw "Not found: $App" }
$appRaw = Get-Content $App -Raw -Encoding UTF8

# 2) Remove any broken one-line injection like:
#    try: ta.add_cross_markers(...) except Exception: pass
$appRaw = [regex]::Replace(
  $appRaw,
  '^[ \t]*try:\s*ta\.add_cross_markers\(.*?\)\s*except\s+Exception:\s*pass[ \t]*\r?\n?',
  '',
  'Multiline'
)

# 3) Ensure the import exists once, after "import streamlit as st"
$hasImport = $appRaw -match '^\s*import\s+ta_overlays\s+as\s+ta\b'
if(-not $hasImport){
  $appRaw = [regex]::Replace(
    $appRaw,
    '^(?<pre>[ \t]*import\s+streamlit\s+as\s+st\s*\r?\n)',
    '${pre}import ta_overlays as ta  # cross markers' + "`r`n",
    'Multiline'
  )
  # Fallback: if streamlit import not found, add at top after future import
  if($appRaw -notmatch '^\s*import\s+ta_overlays\s+as\s+ta\b'){
    $appRaw = [regex]::Replace(
      $appRaw,
      '^(?<head>(?:[ \t]*#.*\r?\n)*[ \t]*from\s+__future__\s+import\s+annotations\s*\r?\n)',
      '${head}import ta_overlays as ta  # cross markers' + "`r`n",
      'Multiline'
    )
  }
}

# 4) Insert a clean multi-line try/except block right before the FIRST st.plotly_chart(
#    and align both the block and st.plotly_chart to the same indent as the nearest opener
#    Find first st.plotly_chart line, capture its current indent
$plotMatch = [regex]::Match($appRaw, '^(?<indent>[ \t]*)st\.plotly_chart\s*\(.*$', [System.Text.RegularExpressions.RegexOptions]::Multiline)
if(-not $plotMatch.Success){ throw "No st.plotly_chart( found." }
$pad = $plotMatch.Groups['indent'].Value

# Build injected block (CRLF line breaks)
$inject = $pad + 'try:' + "`r`n" +
          $pad + '    ta.add_cross_markers(fig, df, price_col="Close", s_short="SMA20", s_long="SMA200")' + "`r`n" +
          $pad + 'except Exception:' + "`r`n" +
          $pad + '    pass' + "`r`n"

# Prepend the injected block to the plotly line using a Replace evaluator
$appRaw = [regex]::Replace(
  $appRaw,
  '^(?<indent>[ \t]*)st\.plotly_chart\s*\((?<rest>.*)$',
  { param($m)
      $padHere = $m.Groups['indent'].Value
      $injectHere = $padHere + 'try:' + "`r`n" +
                    $padHere + '    ta.add_cross_markers(fig, df, price_col="Close", s_short="SMA20", s_long="SMA200")' + "`r`n" +
                    $padHere + 'except Exception:' + "`r`n" +
                    $padHere + '    pass' + "`r`n"
      return $injectHere + $padHere + 'st.plotly_chart(' + $m.Groups['rest'].Value
  },
  [System.Text.RegularExpressions.RegexOptions]::Multiline
)

# 5) Save back
$appRaw | Set-Content -Encoding UTF8 $App

# 6) Restart app
Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
