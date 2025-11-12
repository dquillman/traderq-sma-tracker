# add-version.ps1
$Path = "G:\Users\daveq\traderq\app.py"
$Backup = "G:\Users\daveq\traderq\app.py.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"

if (-not (Test-Path $Path)) {
  Write-Error "app.py not found at $Path"
  exit 1
}

$txt = Get-Content -Raw -Encoding UTF8 $Path

# 1) Ensure APP_VERSION exists (insert at very top if missing)
if ($txt -notmatch '(?m)^\s*APP_VERSION\s*=\s*".*"') {
  $txt = "APP_VERSION = `"v1.3.0`"`r`n" + $txt
  Write-Host "Added APP_VERSION at top."
}

# 2) Ensure a console print after st.set_page_config(...)
if ($txt -notmatch 'Launching TraderQ SMA Tracker \{APP_VERSION\}') {
  $txt = $txt -replace '(st\.set_page_config\([^\)]*\)\s*)', "`$1`r`nprint(f`"🚀 Launching TraderQ SMA Tracker {APP_VERSION}`")`r`n"
  Write-Host "Added console launch print."
}

# 3) Rewrite st.title(...) to include version
#    Matches st.title("something") or st.title(f"something")
$before = $txt
$txt = [regex]::Replace($txt,
  'st\.title\(\s*(?:f)?\"([^\"]*)\"\s*\)',
  { param($m) "st.title(f`"$($m.Groups[1].Value) — {APP_VERSION}`")" },
  'IgnoreCase'
)
if ($txt -ne $before) {
  Write-Host "Replaced st.title(...) with versioned title."
} else {
  Write-Host "No st.title(...) found to replace (already versioned?)."
}

# Save backup then write file
Copy-Item $Path $Backup
$txt | Out-File -Encoding UTF8 $Path
Write-Host "✅ Patched. Backup saved to $Backup"
