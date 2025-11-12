# fix-cross-indent2.ps1 — reinsert cross markers aligned to the nearest block opener
param(
  [string]$App = "G:\Users\daveq\traderq\app.py",
  [int]$Port = 8630
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path $App)) { throw "Not found: $App" }

function Get-Indent([string]$s){
  if ($null -eq $s) { return 0 }
  $t = ($s -replace "`t","    ")
  return ($t.Length - $t.TrimStart().Length)
}

# Read as array of lines
$L = Get-Content $App -Encoding UTF8

# Find first st.plotly_chart(
$plotLineNums = ($L | Select-String -Pattern 'st\.plotly_chart\s*\(').LineNumber
if (!$plotLineNums) { throw "No st.plotly_chart( found in $App" }
$plotIdx = [int]$plotLineNums[0] - 1

# Find the nearest prior block opener (line ending in ':', excluding comments/blank)
$scanStart = [Math]::Max(0, $plotIdx - 80)
$openerIdx = $null
for ($i = $plotIdx - 1; $i -ge $scanStart; $i--) {
  $trim = $L[$i].Trim()
  if ($trim -ne "" -and $trim -notmatch '^#' -and $trim.EndsWith(':')) {
    $openerIdx = $i
    break
  }
}
# Compute target indent: opener indent + 4; if none found, fall back to plot line's current indent
$target = if ($openerIdx -ne $null) { (Get-Indent $L[$openerIdx]) + 4 } else { [Math]::Max(4, (Get-Indent $L[$plotIdx])) }

# Remove any previously injected lines between opener and plot that match our signatures
$startClean = if ($openerIdx -ne $null) { $openerIdx + 1 } else { [Math]::Max(0, $plotIdx - 20) }
for ($i = $plotIdx - 1; $i -ge $startClean; $i--) {
  $t = $L[$i]
  if ($t -match 'ta\.add_cross_markers|^\s*try:\s*$|^\s*except\s+Exception:\s*$|^\s*pass\s*$') {
    $L[$i] = $null
  }
}

# Compact after deletions
$L = $L | Where-Object { $_ -ne $null }

# Recompute plotIdx
$plotLineNums = ($L | Select-String -Pattern 'st\.plotly_chart\s*\(').LineNumber
$plotIdx = [int]$plotLineNums[0] - 1

# Build clean injected block with correct indentation
$pad  = ' ' * $target
$pad2 = ' ' * ($target + 4)
$inject = @(
  "$pad" + "try:",
  "$pad2" + "ta.add_cross_markers(fig, df, price_col=""Close"", s_short=""SMA20"", s_long=""SMA200"")",
  "$pad" + "except Exception:",
  "$pad2" + "pass"
)

# Insert block immediately BEFORE st.plotly_chart
$before = if ($plotIdx -gt 0) { $L[0..($plotIdx-1)] } else { @() }
$after  = $L[$plotIdx..($L.Count-1)]
$L = @()
$L += $before
$L += $inject
$L += $after

# Normalize st.plotly_chart line to the same target indent
$plotLineNums = ($L | Select-String -Pattern 'st\.plotly_chart\s*\(').LineNumber
$plotIdx = [int]$plotLineNums[0] - 1
$L[$plotIdx] = (' ' * $target) + $L[$plotIdx].TrimStart()

# Ensure the import exists once (after streamlit import)
$lines = $L
$lines = $lines | Where-Object { $_ -notmatch '^\s*import\s+ta_overlays\b' }
$stLineNums = ($lines | Select-String -Pattern '^\s*import\s+streamlit\s+as\s+st\s*$').LineNumber
if ($stLineNums) {
  $i = [int]$stLineNums[0] - 1
  $pre = $lines[0..$i]
  $post = if ($i + 1 -lt $lines.Count) { $lines[($i+1)..($lines.Count-1)] } else { @() }
  $lines = @(); $lines += $pre; $lines += 'import ta_overlays as ta  # cross markers'; $lines += $post
} else {
  $lines = @('import ta_overlays as ta  # cross markers') + $lines
}

# Write back
$lines | Set-Content -Path $App -Encoding UTF8

# Restart the app
Get-Process streamlit,python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$venvPy = "G:\Users\daveq\traderq\.venv\Scripts\python.exe"
& $venvPy -m streamlit run $App --server.port $Port
