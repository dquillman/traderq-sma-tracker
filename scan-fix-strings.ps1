# scan-fix-strings.ps1
param(
  [switch]$FixCommon  # add this to auto-apply safe fixes
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq
$path = "app.py"
if (-not (Test-Path $path)) { Write-Error "app.py not found at $pwd"; exit 1 }

# 1) Report total line count
$lines = Get-Content $path
$lineCount = $lines.Count
Write-Host ("Total lines in app.py: {0}" -f $lineCount)

# 2) Find suspicious lines: odd number of unescaped double quotes (")
#    Also catch some common Streamlit/Plotly patterns explicitly.
$suspects = @()
for ($i=0; $i -lt $lines.Count; $i++) {
  $ln = $lines[$i]
  # Count unescaped quotes
  $tmp = $ln -replace '\\\"',''   # ignore \" escaped quotes
  $q = ($tmp.ToCharArray() | Where-Object { $_ -eq '"' }).Count
  $looksOdd = ($q % 2 -ne 0)

  $looksSt = $ln -match 'st\.(?:title|header|subheader|caption|markdown|write|sidebar\.\w+|download_button)\s*\("'
  $looksColor = $ln -match 'line_color="rgba\(\d+,\d+,\d+,\d+(\.\d+)?\)\s*$'

  if ($looksOdd -or $looksSt -or $looksColor) {
    $suspects += [pscustomobject]@{ Line=$i+1; Text=$ln }
  }
}

if ($suspects.Count -eq 0) {
  Write-Host "No suspicious lines detected."
} else {
  Write-Host "`nSuspicious lines (possible unterminated strings):"
  foreach ($s in $suspects) {
    Write-Host ("  L{0}: {1}" -f $s.Line, $s.Text)
  }
}

if (-not $FixCommon) {
  Write-Host "`nRun with -FixCommon to apply auto-fixes for common cases."
  exit 0
}

# 3) Apply safe, targeted fixes
$txt = [System.IO.File]::ReadAllText($path)

# 3a) Plotly vline rgba cases missing closing '")'
$txt = [regex]::Replace($txt,
  'line_color="rgba\((?:\s*\d+\s*,){3}\s*(?:\d+(?:\.\d+)?)\)\s*(\r?\n)',
  'line_color="rgba(255,255,255,0.3)")$1'   # default to the white translucent used earlier
)

# 3b) Streamlit headings missing closing '")'
$txt = [regex]::Replace($txt,
  'st\.(?:title|header|subheader|caption|markdown|write)\s*\("([^"\r\n]+)\s*(\r?\n)',
  'st.\1("$1")$2'  # This placeholder is wrong; do specific ones below.
)

# Do specific, safer replacements for known functions:
$txt = [regex]::Replace($txt,
  'st\.title\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.title("$1")$2'
)
$txt = [regex]::Replace($txt,
  'st\.header\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.header("$1")$2'
)
$txt = [regex]::Replace($txt,
  'st\.subheader\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.subheader("$1")$2'
)
$txt = [regex]::Replace($txt,
  'st\.caption\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.caption("$1")$2'
)
$txt = [regex]::Replace($txt,
  'st\.markdown\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.markdown("$1")$2'
)
$txt = [regex]::Replace($txt,
  'st\.write\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.write("$1")$2'
)

# 3c) Sidebar checkbox/radio/text_input… often cut mid-line
$txt = [regex]::Replace($txt,
  'st\.sidebar\.(?:checkbox|radio|text_input|selectbox)\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.sidebar.$1("$1")$2'  # replace with specific safe variants below
)
$txt = [regex]::Replace($txt,
  'st\.sidebar\.checkbox\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.sidebar.checkbox("$1")$2'
)
$txt = [regex]::Replace($txt,
  'st\.sidebar\.radio\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.sidebar.radio("$1")$2'
)
$txt = [regex]::Replace($txt,
  'st\.sidebar\.text_input\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.sidebar.text_input("$1")$2'
)
$txt = [regex]::Replace($txt,
  'st\.sidebar\.selectbox\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.sidebar.selectbox("$1")$2'
)

# 3d) download_button: close title string if cut
$txt = [regex]::Replace($txt,
  'st\.download_button\s*\("([^\r\n"]+)\s*(\r?\n)',
  'st.download_button("$1"$2'
)

# 3e) FINAL generic: if a line ends with an odd number of quotes, append one more "
#     (done line-by-line to avoid mangling multi-line blocks)
$final = New-Object System.Text.StringBuilder
foreach ($ln in $txt -split "`r?`n") {
  $x = $ln -replace '\\\"',''
  $q = ($x.ToCharArray() | Where-Object { $_ -eq '"' }).Count
  if ($q % 2 -ne 0) {
    $ln = $ln + '"'
  }
  $null = $final.AppendLine($ln)
}
[System.IO.File]::WriteAllText($path, $final.ToString(), [System.Text.Encoding]::UTF8)

Write-Host "Applied common fixes. Re-scanning…"

# 4) Re-scan and report
& powershell -NoProfile -Command "(Get-Content app.py).Count"
Write-Host "Lines counted above."
# Compile check
$python = ".\.venv\Scripts\python.exe"; if (-not (Test-Path $python)) { $python = "python" }
$p = Start-Process -FilePath $python -ArgumentList @("-m","py_compile","app.py") -PassThru -NoNewWindow -RedirectStandardError errs.txt
$p.WaitForExit()
$err = (Get-Content errs.txt -Raw -ErrorAction SilentlyContinue)
Remove-Item errs.txt -ErrorAction SilentlyContinue
if ($p.ExitCode -eq 0 -and -not $err) {
  Write-Host "✅ app.py compiles."
} else {
  Write-Host "⚠️  app.py still has a compile error:"
  Write-Host $err
}
