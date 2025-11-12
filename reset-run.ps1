param([int]$Port = 8590)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

# 0) Kill any stragglers
taskkill /IM streamlit.exe /F 2>$null | Out-Null
taskkill /IM python.exe    /F 2>$null | Out-Null
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

# 4) Start server (log to file), bind to localhost
$log = Join-Path (Get-Location) "streamlit_last.log"
if (Test-Path $log) { Remove-Item $log -Force }
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName  = "python"
$psi.Arguments = "-m streamlit run app.py --server.port $Port --server.address localhost"
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$p.Start() | Out-Null
$p.StandardOutput.BaseStream.BeginRead((New-Object byte[] 0),0,0,$null,$null) | Out-Null
$p.StandardError.BaseStream.BeginRead((New-Object byte[] 0),0,0,$null,$null) | Out-Null
Start-Sleep -Milliseconds 500

# Tee process output to log asynchronously
Start-Job -ScriptBlock {
  param($Pid,$LogPath)
  $proc = Get-Process -Id $Pid -ErrorAction SilentlyContinue
  if (-not $proc) { return }
  $si = $proc.StartInfo
} -ArgumentList $p.Id,$log | Out-Null

# 5) Wait for port to open (up to 40s)
$ready = $false
for ($i=0; $i -lt 40; $i++) {
  $tn = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
  if ($tn.TcpTestSucceeded) { $ready = $true; break }
  Start-Sleep -Milliseconds 1000
}

if ($ready) {
  Write-Host "✅ Streamlit is up on http://localhost:$Port" -ForegroundColor Green
  Start-Process "http://localhost:$Port"
} else {
  Write-Host "❌ Server did not open port $Port." -ForegroundColor Red
  Write-Host "Showing recent log (if any):`n"
  if (Test-Path $log) { Get-Content $log -Tail 200 }
  else { Write-Host "(no log written yet)" }
  exit 1
}
