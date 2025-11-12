# fix-cross-indent.ps1 — normalize indentation around cross-markers and st.plotly_chart
param(
  [string]$App = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)

$ErrorActionPreference = 'Stop'

if (!(Test-Path $App)) { throw "Not found: $App" }

# Read file as lines
$L = Get-Content $App -Encoding UTF8

# Helper: count leading spaces (tabs count as 4 spaces)
function Get-Indent([string]$s){
  if ($null -eq $s) { return 0 }
  $t = ($s -replace "`t","    ")
  return ($t.Length - $t.TrimStart().Length)
}

# Locate the first st.plotly_chart(
$plotIdx = ($L | Select-String -Pattern 'st\.plotly_chart\s*\(').LineNumber
if (-not $plotIdx) { throw "Could not find st.plotly_chart( in $App" }
$plotIdx = [int]$plotIdx[0] - 1

# Search up to 20 lines above for the injected try:/except/pass and our ta.add_cross_markers call
$startSearch = [Math]::Max(0, $plotIdx - 20)
$tryIdx = $null
$exceptIdx = $null
$addIdx = $null
$passIdx = $null

for ($i=$plotIdx-1; $i -ge $startSearch; $i--) {
  if ($null -eq $tryIdx     -and $L[$i] -match '^\s*try:\s*$') { $tryIdx = $i }
  if ($null -eq $exceptIdx  -and $L[$i] -match '^\s*except\s+Exception:\s*$') { $exceptIdx = $i }
  if ($null -eq $passIdx    -and $L[$i] -match '^\s*pass\s*$') { $passIdx = $i }
  if ($null -eq $addIdx     -and $L[$i] -match 'ta\.add_cross_markers\s*\(') { $addIdx = $i }
  if ($tryIdx -and $exceptIdx -and $addIdx -and $passIdx) { break }
}

# Determine base indentation: align everything to the try: indent if present; else align to current plot indent or 8 spaces
$baseIndent = 8
if ($tryIdx -ne $null) { $baseIndent = Get-Indent $L[$tryIdx] }
elseif ($plotIdx -ne $null) { $baseIndent = Get-Indent $L[$plotIdx] }

# Normalize: ensure the block exists; if some parts are missing, create them right before st.plotly_chart
$insertBlock = @()
$insertBlock += (' ' * $baseIndent)     + 'try:'
$insertBlock += (' ' * ($baseIndent+4)) + 'ta.add_cross_markers(fig, df, price_col="Close", s_short="SMA20", s_long="SMA200")'
$insertBlock += (' ' * $baseIndent)     + 'except Exception:'
$insertBlock += (' ' * ($baseIndent+4)) + 'pass'

# Remove any existing ta.add_cross_markers/try/except/pass directly above the plot call (to avoid duplicates)
for ($i=$plotIdx-1; $i -ge $startSearch; $i--) {
  if ($L[$i] -match 'ta\.add_cross_markers|^\s*try:\s*$|^\s*except\s+Exception:\s*$|^\s*pass\s*$') {
    $L[$i] = $null
  } else {
    # stop when we hit a line that is not part of our injected block
    if ($L[$i] -ne $null -and $L[$i].Trim() -ne '') { break }
  }
}

# Compact lines after nulling
$L = $L | Where-Object { $_ -ne $null }

# Recompute plot index after compaction
$plotIdx = ($L | Select-String -Pattern 'st\.plotly_chart\s*\(').LineNumber
$plotIdx = [int]$plotIdx[0] - 1

# Insert the clean block immediately before st.plotly_chart
$before = if ($plotIdx -gt 0) { $L[0..($plotIdx-1)] } else { @() }
$after  = $L[$plotIdx..($L.Count-1)]
$L = @()
$L += $before
$L += $insertBlock
$L += $after

# Finally, ensure st.plotly_chart line uses baseIndent (not over-indented)
$plotIdx = ($L | Select-String -Pattern 'st\.plotly_chart\s*\(').LineNumber
$plotIdx = [int]$plotIdx[0] - 1
$plotLine = $L[$plotIdx].TrimStart()
$L[$plotIdx] = (' ' * $baseIndent) + $plotLine

# Save
$L | Set-Content -Path $App -Encoding UTF8

# Restart the app so you can see the result immediately
Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = "G:\Users\daveq\traderq\.venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
