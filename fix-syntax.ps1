# fix-syntax.ps1
Set-Location G:\Users\daveq\traderq
$src = "app.py"
$bak = "$src.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $src $bak -Force
Write-Host "[OK] Backup saved to $bak" -ForegroundColor Green

# Read and patch
$content = Get-Content $src -Raw
# Remove accidental double quotes and stray )"
$content = $content -replace '\)\s*"\)', ')'
$content = $content -replace '"\)\s*$', ')'

Set-Content $src $content -Encoding UTF8
Write-Host "[OK] Repaired unterminated string quotes in app.py" -ForegroundColor Green
