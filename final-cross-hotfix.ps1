# final-cross-hotfix.ps1 — fix caption vars, ensure counts exist, restart
param(
  [string]$App = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'
$App = [System.IO.Path]::GetFullPath($App)
if (!(Test-Path $App)) { throw "Not found: $App" }

# 1) Read file
$text = Get-Content $App -Raw -Encoding UTF8

# 2) Fix caption to use _golden/_death and literal SMA names (no s20/s200 vars)
$text = $text -replace 'golden=\{gu\}', 'golden={_golden}'
$text = $text -replace 'death=\{gd\}', 'death={_death}'
$text = $text -replace 's20=\{s20\}', 's20=SMA20'
$text = $text -replace 's200=\{s200\}', 's200=SMA200'

# 3) Ensure the count variables exist.
#    Insert right AFTER the add_cross_markers(...) line nearest the first st.plotly_chart(, aligned to its indent.
$plot = [regex]::Match($text,'^(?<indent>[ \t]*)st\.plotly_chart\s*\(.*$', 'Multiline')
if (-not $plot.Success) { throw "Could not find st.plotly_chart(" }
$plotIndent = $plot.Groups['indent'].Value

# If counts are missing, inject them after the add_cross_markers(...) that precedes the plot call.
if ($text -notmatch '_golden\s*=\s*int\(') {
  # Find the add_cross_markers line before the plot
  $beforePlot = $text.Substring(0, $plot.Index)
  $markerCall = [regex]::Matches($beforePlot, '^(?<indent>[ \t]*)add_cross_markers\s*\(.*$', 'Multiline') | Select-Object -Last 1
  if ($markerCall) {
    $indent = $markerCall.Groups['indent'].Value
    $inject = @(
      "$indent`_s_prev = df[`"SMA20`"].shift(1); _l_prev = df[`"SMA200`"].shift(1)",
      "$indent`_golden = int(((_s_prev -le _l_prev) -and (df[`"SMA20`"] -gt df[`"SMA200`"])).sum())",
      "$indent`_death  = int(((_s_prev -ge _l_prev) -and (df[`"SMA20`"] -lt df[`"SMA200`"])).sum())"
    ) -join "`r`n"

    # Insert after the marker call line
    $start = $markerCall.Index + $markerCall.Length
    $text = $text.Substring(0,$start) + "`r`n" + $inject + $text.Substring($start)
  } else {
    # Fallback: put counts right before the plot line at the same indent as the plot block
    $indent = $plotIndent
    $inject = @(
      "$indent`_s_prev = df[`"SMA20`"].shift(1); _l_prev = df[`"SMA200`"].shift(1)",
      "$indent`_golden = int(((_s_prev -le _l_prev) -and (df[`"SMA20`"] -gt df[`"SMA200`"])).sum())",
      "$indent`_death  = int(((_s_prev -ge _l_prev) -and (df[`"SMA20`"] -lt df[`"SMA200`"])).sum())"
    ) -join "`r`n"
    $text = $text.Substring(0,$plot.Index) + $inject + "`r`n" + $text.Substring($plot.Index)
  }
}

# 4) Write back (UTF-8 no BOM)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($App, $text, $utf8NoBom)

# 5) Restart Streamlit
Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = Join-Path (Split-Path $App -Parent) ".venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
