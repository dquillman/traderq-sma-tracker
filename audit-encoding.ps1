# audit-encoding.ps1 — find & optionally fix non-ASCII across repo (safe ASCII-only script)
param(
  [switch]$Fix,           # if set, attempt safe replacements + strip BOMs + save UTF-8 (no BOM)
  [switch]$KeepUnicode,   # with -Fix, keep pretty Unicode (— “ ” …) instead of ASCII fallbacks
  [int]$Port = 8620,      # optional: run app after fixes
  [switch]$Run            # start Streamlit after fixes
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

function New-UTF8NoBOM { param($s,$path)
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($path, $s, $enc)
}

function Remove-BOM { param($path)
  $bytes = [System.IO.File]::ReadAllBytes($path)
  if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    [System.IO.File]::WriteAllBytes($path, $bytes[3..($bytes.Length-1)])
  }
}

# Build strings from codepoints (no non-ASCII literals in this script)
function CP([int]$hex){ [char]$hex }

# Common Unicode punctuation
$EM   = CP 0x2014  # —
$EN   = CP 0x2013  # –
$LSQ  = CP 0x2018  # ‘
$RSQ  = CP 0x2019  # ’
$LDQ  = CP 0x201C  # “
$RDQ  = CP 0x201D  # ”
$ELL  = CP 0x2026  # …
$BULL = CP 0x2022  # •
$MDOT = CP 0x00B7  # ·
$LGT  = CP 0x26A1  # ⚡

# Map: Unicode -> ASCII
$U2A = @{
  $EM   = "--";  $EN   = "-";
  $LSQ  = "'";   $RSQ  = "'";
  $LDQ  = '"';   $RDQ  = '"';
  $ELL  = "..."; $BULL = "*";
  $MDOT = ".";   $LGT  = "Lightning";
}

# Map: fix mojibake sequences -> ASCII
$M2A = @{
  "â€”"="--"; "â€“"="-"; "â€˜"="'"; "â€™"="'"; "â€œ"='"'; "â€"='"';
  "â€¦"="..."; "â€¢"="*"; "âš¡"="Lightning"; "Â·"="."; "Â"="";
}

# Map: fix mojibake sequences -> proper Unicode (if -KeepUnicode)
$M2U = @{
  "â€”"=$EM; "â€“"=$EN; "â€˜"=$LSQ; "â€™"=$RSQ; "â€œ"=$LDQ; "â€"=$RDQ;
  "â€¦"=$ELL; "â€¢"=$BULL; "âš¡"=$LGT; "Â·"=$MDOT; "Â"="";
}

# Which files to scan
$files = Get-ChildItem -Recurse -File -Include *.py,*.toml,*.md,*.css
if (-not $files) { Write-Host "No files to scan."; exit 0 }

$offenders = @()
foreach ($f in $files) {
  $text = [System.IO.File]::ReadAllText($f.FullName, [System.Text.Encoding]::UTF8)
  $lines = $text -split "`r?`n", [System.StringSplitOptions]::None
  for ($i=0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    if ($line.ToCharArray() | Where-Object { [int]$_ -gt 127 }) {
      # Collect a compact hex preview of non-ASCII chars on this line
      $badChars = ($line.ToCharArray() | Where-Object { [int]$_ -gt 127 } | ForEach-Object { "U+" + ([int]$_).ToString("X4") }) -join ","
      $offenders += [PSCustomObject]@{ File=$f.FullName; Line=$i+1; Preview=$line.Trim(); Codes=$badChars }
    }
  }
}

if ($offenders.Count -gt 0) {
  Write-Host "Found non-ASCII in $($offenders.Count) line(s):" -ForegroundColor Yellow
  $offenders | Format-Table -AutoSize
} else {
  Write-Host "No non-ASCII found." -ForegroundColor Green
}

if (-not $Fix) { exit 0 }

# FIX MODE
$changed = 0
foreach ($f in $files) {
  $orig = [System.IO.File]::ReadAllText($f.FullName, [System.Text.Encoding]::UTF8)
  $txt  = $orig

  if ($KeepUnicode) {
    foreach ($k in $M2U.Keys) { $txt = $txt.Replace($k, $M2U[$k]) }
    # Save UTF-8 no BOM
    if ($txt -ne $orig) {
      Copy-Item $f.FullName "$($f.FullName).bak_$(Get-Date -Format yyyyMMdd_HHmmss)" -Force
      New-UTF8NoBOM $txt $f.FullName
      $changed++
    } else {
      # Even if not changed, strip BOM if present
      Remove-BOM $f.FullName
    }
  } else {
    # Down-convert everything to ASCII-safe
    foreach ($k in $U2A.Keys) { $txt = $txt.Replace($k, $U2A[$k]) }
    foreach ($k in $M2A.Keys) { $txt = $txt.Replace($k, $M2A[$k]) }
    $txt = [System.Text.RegularExpressions.Regex]::Replace($txt, '[^\u0000-\u007F]', '')
    if ($txt -ne $orig) {
      Copy-Item $f.FullName "$($f.FullName).bak_$(Get-Date -Format yyyyMMdd_HHmmss)" -Force
      # Save ASCII (or UTF-8 no BOM if you prefer)
      [System.IO.File]::WriteAllText($f.FullName, $txt, [System.Text.Encoding]::ASCII)
      $changed++
    } else {
      Remove-BOM $f.FullName
    }
  }
}

Write-Host "[OK] Fixed $changed file(s)." -ForegroundColor Green

# Optional: quick Python syntax check for all .py
$pyFiles = Get-ChildItem -Recurse -File -Include *.py | Select-Object -Expand FullName
if ($pyFiles) {
  try {
    . .\.venv\Scripts\Activate.ps1 2>$null | Out-Null
  } catch {}
  foreach ($p in $pyFiles) {
    & python - <<'PY'
import sys, py_compile
try:
    py_compile.compile(sys.argv[1], doraise=True)
    print("[OK] Syntax:", sys.argv[1])
except Exception as e:
    print("[ERR] Syntax:", sys.argv[1], "->", e)
    sys.exit(1)
PY
    if ($LASTEXITCODE -ne 0) { Write-Host "Syntax error in $p" -ForegroundColor Red; exit 1 }
  }
}

if ($Run -and (Test-Path .\app.py)) {
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  . .\.venv\Scripts\Activate.ps1 2>$null | Out-Null
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}
