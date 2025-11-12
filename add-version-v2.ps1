# add-version-v2.ps1  (ASCII-safe)
$Path   = "G:\Users\daveq\traderq\app.py"
$Backup = "G:\Users\daveq\traderq\app.py.bak_{0}" -f (Get-Date -Format yyyyMMdd_HHmmss)

if (-not (Test-Path $Path)) {
  Write-Error "app.py not found at $Path"
  exit 1
}

# Read whole file
$txt = Get-Content -Raw -Encoding UTF8 $Path
$orig = $txt

# 1) Ensure APP_VERSION exists at top
if ($txt -notmatch '(?m)^\s*APP_VERSION\s*=\s*".*"') {
  $txt = 'APP_VERSION = "v1.3.0"' + "`r`n" + $txt
  Write-Host "[+] Added APP_VERSION at top."
}

# 2) Ensure console print after st.set_page_config(...)
if ($txt -notmatch 'Launching TraderQ SMA Tracker') {
  # Insert a print right after the first st.set_page_config(...) call
  $txt = [regex]::Replace($txt, '(st\.set_page_config\([^\)]*\))', '$1' + "`r`n" + 'print(f"Launching TraderQ SMA Tracker {APP_VERSION}")', 1)
  Write-Host "[+] Added console launch print."
}

# 3) Replace the FIRST st.title(...) with a versioned title (ASCII hyphen)
#    We deliberately overwrite the first title line to avoid complex quoting.
if ($txt -match 'st\.title\([^\)]*\)') {
  $txt = [regex]::Replace($txt, 'st\.title\([^\)]*\)', 'st.title(f"TraderQ SMA 20/200 Tracker - {APP_VERSION}")', 1)
  Write-Host "[+] Replaced first st.title(...) with versioned title."
} else {
  Write-Host "[i] No st.title(...) found to replace."
}

# Only write if changed
if ($txt -ne $orig) {
  Copy-Item $Path $Backup -ErrorAction SilentlyContinue
  $txt | Out-File -Encoding UTF8 $Path
  Write-Host "[OK] Patched. Backup => $Backup"
} else {
  Write-Host "[i] No changes made (already patched?)."
}
