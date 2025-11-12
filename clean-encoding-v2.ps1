param(
  [switch]$KeepUnicode,   # If set, keep nice Unicode punctuation; else down-convert to ASCII
  [int]$Port = 8617
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

function OK($m){ Write-Host "[OK] $m" -ForegroundColor Green }
function Info($m){ Write-Host "[i] $m" -ForegroundColor Cyan }

# Helper: build strings from codepoints without embedding non-ASCII in the script
function CharFromHex([int]$hex){ return [char]$hex }

# Common punctuation (Unicode codepoints)
$EM     = CharFromHex 0x2014   # —
$EN     = CharFromHex 0x2013   # –
$LSQ    = CharFromHex 0x2018   # ‘
$RSQ    = CharFromHex 0x2019   # ’
$LDQ    = CharFromHex 0x201C   # “
$RDQ    = CharFromHex 0x201D   # ”
$ELL    = CharFromHex 0x2026   # …
$BULL   = CharFromHex 0x2022   # •
$MDOT   = CharFromHex 0x00B7   # ·
$LGT    = CharFromHex 0x26A1   # ⚡

# Replacement maps
# A) Unicode -> ASCII fallbacks (default)
$unicodeToAscii = @{
  $EM   = "--"
  $EN   = "-"
  $LSQ  = "'"
  $RSQ  = "'"
  $LDQ  = '"'
  $RDQ  = '"'
  $ELL  = "..."
  $BULL = "*"
  $MDOT = "."
  $LGT  = "Lightning"
}

# B) Mojibake patterns (these are sequences of Latin-1 chars; avoid non-ASCII by hex)
# We'll detect typical broken UTF-8 sequences and down-convert to ASCII equivalents.
$mojiSeqs = @(
  @{ Bad = "â€”"; Fix = "--" },  # em dash
  @{ Bad = "â€“"; Fix = "-"  },  # en dash
  @{ Bad = "â€˜"; Fix = "'"  },  # left single
  @{ Bad = "â€™"; Fix = "'"  },  # right single
  @{ Bad = "â€œ"; Fix = '"'  },  # left double
  @{ Bad = "â€"; Fix = '"'  },  # right double (variant)
  @{ Bad = "â€¦"; Fix = "..."},  # ellipsis
  @{ Bad = "â€¢"; Fix = "*"  },  # bullet
  @{ Bad = "âš¡"; Fix = "Lightning" }, # lightning
  @{ Bad = "Â·";  Fix = "."  },  # middle dot variant
  @{ Bad = "Â";   Fix = ""   }   # stray "Â"
)

# If the user wants to KEEP Unicode punctuation (and only fix mojibake):
$mojiToUnicode = @{
  "â€”" = $EM
  "â€“" = $EN
  "â€˜" = $LSQ
  "â€™" = $RSQ
  "â€œ" = $LDQ
  "â€" = $RDQ
  "â€¦" = $ELL
  "â€¢" = $BULL
  "âš¡" = $LGT
  "Â·"  = $MDOT
  "Â"   = ""
}

function Replace-Literals([string]$s, [hashtable]$map){
  foreach($k in $map.Keys){ $s = $s.Replace($k, $map[$k]) }
  return $s
}
function Replace-MojiList([string]$s, $list){
  foreach($row in $list){ $s = $s.Replace($row.Bad, $row.Fix) }
  return $s
}

# Files to clean
$files = Get-ChildItem -Recurse -File -Include *.py,*.toml,*.md,*.css
if (-not $files){ Write-Host "No files found to clean."; exit 0 }

$changed = 0
foreach($f in $files){
  # Read raw bytes then UTF-8 decode (so we can detect mojibake textually)
  $raw  = [System.IO.File]::ReadAllBytes($f.FullName)
  $text = [System.Text.Encoding]::UTF8.GetString($raw)
  $orig = $text

  if ($KeepUnicode){
    # Fix mojibake back into proper Unicode punctuation
    $text = Replace-Literals $text $mojiToUnicode
    # Save as UTF-8 without BOM
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    if ($text -ne $orig){
      Copy-Item $f.FullName "$($f.FullName).bak_$(Get-Date -Format yyyyMMdd_HHmmss)" -Force
      [System.IO.File]::WriteAllText($f.FullName, $text, $utf8NoBom)
      $changed++
      Write-Host "Fixed (Unicode) -> $($f.FullName)"
    }
  } else {
    # Default: normalize to ASCII-safe
    $text = Replace-Literals $text $unicodeToAscii     # Real Unicode -> ASCII
    $text = Replace-MojiList $text $mojiSeqs           # Broken sequences -> ASCII
    # Strip anything still non-ASCII
    $text = [System.Text.RegularExpressions.Regex]::Replace($text, '[^\u0000-\u007F]', '')
    if ($text -ne $orig){
      Copy-Item $f.FullName "$($f.FullName).bak_$(Get-Date -Format yyyyMMdd_HHmmss)" -Force
      # Save as ASCII
      $enc = [System.Text.Encoding]::ASCII
      [System.IO.File]::WriteAllText($f.FullName, $text, $enc)
      $changed++
      Write-Host "Fixed (ASCII) -> $($f.FullName)"
    }
  }
}

OK "Cleaned $changed file(s)."

# Relaunch app if present
if (Test-Path .\app.py){
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  if (Test-Path .\.venv\Scripts\Activate.ps1){ . .\.venv\Scripts\Activate.ps1 }
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}
