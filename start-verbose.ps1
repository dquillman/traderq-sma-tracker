param([int]$Port = 8591)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

# 0) Kill any stragglers (quiet)
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name python    -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 300

# 1) Ensure venv
if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
  & "G:\Python311\python.exe" -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

# 2) Deps
if (Test-Path .\requirements.txt) {
  python -m pip install --upgrade pip
  python -m pip install --no-cache-dir -r .\requirements.txt
} else {
  python -m pip install --no-cache-dir streamlit pandas numpy pandas-datareader plotly pycoingecko yfinance
}

# 3) Clear cache
streamlit cache clear | Out-Null

# 4) Run Streamlit synchronously and watch output
$log = Join-Path (Get-Location) "streamlit_verbose.log"
if (Test-Path $log) { Remove-Item $log -Force }

Write-Host "Starting Streamlit on http://localhost:$Port ..." -ForegroundColor Cyan

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName  = "python"
$psi.Arguments = "-m streamlit run app.py --server.port $Port --server.address localhost"
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$p.Start() | Out-Null

# stream output to console + log
$sw = [System.IO.StreamWriter]::new($log, $false, [System.Text.Encoding]::UTF8)
$opened = $false
while (-not $p.HasExited) {
  while (-not $p.StandardOutput.EndOfStream) {
    $line = $p.StandardOutput.ReadLine()
    $sw.WriteLine($line)
    Write-Host $line
    if (-not $opened -and $line -match 'Local URL:\s*(http://[^\s]+)') {
      $url = $Matches[1]
      Start-Process $url
      $opened = $true
    }
  }
  while (-not $p.StandardError.EndOfStream) {
    $eline = $p.StandardError.ReadLine()
    $sw.WriteLine($eline)
    Write-Host $eline -ForegroundColor Yellow
  }
  Start-Sleep -Milliseconds 100
}
$sw.Flush(); $sw.Close()

if ($opened) {
  Write-Host "`n✅ Streamlit started and browser opened." -ForegroundColor Green
  exit 0
} else {
  Write-Host "`n❌ Streamlit did not expose a Local URL. Showing last 200 log lines:" -ForegroundColor Red
  if (Test-Path $log) { Get-Content $log -Tail 200 }
  exit 1
}
