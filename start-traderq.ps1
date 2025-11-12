# bump-and-push.ps1
param([string]$Msg = "chore: bump version & sync")

$ErrorActionPreference = 'Stop'
cd G:\Users\daveq\traderq

if (-not (Test-Path .\app.py)) { throw "app.py not found" }

$txt = Get-Content -Raw -Encoding UTF8 .\app.py
if ($txt -match 'APP_VERSION\s*=\s*"v(\d+)\.(\d+)\.(\d+)"') {
  $maj=[int]$Matches[1]; $min=[int]$Matches[2]; $pat=[int]$Matches[3]+1
  $new = "APP_VERSION = `"v$maj.$min.$pat`""
  $txt = [regex]::Replace($txt,'APP_VERSION\s*=\s*".*?"',$new,1)
  $txt | Set-Content -Encoding UTF8 .\app.py
  Write-Host "Bumped to v$maj.$min.$pat"
} else {
  throw "APP_VERSION not found in app.py"
}

# ensure requirements.txt contains all deps
$req = @(
  'streamlit','pandas','numpy','pandas-datareader','plotly','pycoingecko','yfinance'
)
$reqPath = ".\requirements.txt"
$existing = @()
if (Test-Path $reqPath) { $existing = (Get-Content $reqPath | ForEach-Object { $_.Trim() }) }
$merged = ($existing + $req) | Where-Object { $_ -ne "" } | Select-Object -Unique
$merged | Out-File -Encoding UTF8 $reqPath
Write-Host "requirements.txt updated"

git add app.py requirements.txt
git commit -m $Msg
git push -u origin main
