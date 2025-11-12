param(
  [switch]$KeepUnicode,   # if set, keep proper Unicode; otherwise force ASCII-safe output
  [int]$Port = 8617
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

function OK($m){ Write-Host "[OK] $m" -ForegroundColor Green }
function Info($m){ Write-Host "[i] $m" -ForegroundColor Cyan }

# Files to scan
$files = Get-ChildItem -Recurse -File -Include *.py,*.toml,*.md,*.css

if (-not $files) {
  Write-Host "No files found to clean." -ForegroundColor Yellow
  exit 0
}

# Map helpers (literal replacements, not regex)
function Apply-ReplacementsLiteral {
  param([string]$s, [hashtable]$map)
  foreach ($k in $map.Keys) { $s = $s.Replace($k, $map[$k]) }
  return $s
}

# Mojibake -> proper Unicode (used when -KeepUnicode)
$mojiToUnicode = @{
  'â€”' = '—';  # em dash
  'â€“' = '–';  # en dash
  'â€˜' = '‘';  # left single
  'â€™' = '’';  # right single
  'â€œ' = '“';  # left double
  'â€' = '”';  # right double (note some shells show as â€)
  'â€¦' = '…';  # ellipsis
  'â€¢' = '•';  # bullet
  'âš¡' = '⚡';  # lightning
  'Â·'  = '·';  # middle dot
  'Â'   = ''    # stray control from cp1252 mix
}

# Unicode -> ASCII-safe (used by default)
$unicodeToAscii = @{
  '—' = '--'
  '–' = '-'
  '‘' = "'"
  '’' = "'"
  '‚' = "'"
  '“' = '"'
  '”' = '"'
  '„' = '"'
  '…' = '...'
  '•' = '*'
  '·' = '.'
  '⚡' = 'Lightning'  # keep it readable in ASCII mode
}

# Mojibake -> ASCII fallbacks (covers when file already garbled)
$mojiToAscii = @{
  'â€”' = '--'
  'â€“' = '-'
  'â€˜' = "'"
  'â€™' = "'"
  'â€œ' = '"'
  'â€' = '"'
  'â€¦' = '...'
  'â€¢' = '*'
  'âš¡' = 'Lightning'
  'Ã—'  = 'x'
  'Â·'  = '.'
  'Â'   = ''
}

# Process each file
$changed = 0
foreach ($f in $files) {
  $raw = Get-Content $f.FullName -Raw -Encoding Byte
  # Try interpret as UTF-8 first
  $text = [System.Text.Encoding]::UTF8.GetString($raw)

  $orig = $text

  if ($KeepUnicode) {
    # Fix mojibake back to Unicode
    $text = Apply-ReplacementsLiteral $text $mojiToUnicode
    # Save as UTF-8 without BOM
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    if ($text -ne $orig) {
      Copy-Item $f.FullName "$($f.FullName).bak_$(Get-Date -Format yyyyMMdd_HHmmss)" -Force
      [System.IO.File]::WriteAllText($f.FullName, $text, $utf8NoBom)
      $changed++
      Write-Host "Fixed (Unicode) -> $($f.FullName)"
    }
  } else {
    # ASCII-safe mode: convert Unicode punctuation to ASCII, and fix mojibake to ASCII
    $text = Apply-ReplacementsLiteral $text $unicodeToAscii
    $text = Apply-ReplacementsLiteral $text $mojiToAscii
    # Strip any remaining non-ASCII
    $text = [System.Text.RegularExpressions.Regex]::Replace($text, '[^\u0000-\u007F]', '')

    if ($text -ne $orig) {
      Copy-Item $f.FullName "$($f.FullName).bak_$(Get-Date -Format yyyyMMdd_HHmmss)" -Force
      # Save as ASCII
      $ascii = [System.Text.Encoding]::ASCII
      [System.IO.File]::WriteAllText($f.FullName, $text, $ascii)
      $changed++
      Write-Host "Fixed (ASCII) -> $($f.FullName)"
    }
  }
}

OK "Cleaned $changed file(s)."

# Relaunch Streamlit (only if app.py exists)
if (Test-Path .\app.py) {
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}
