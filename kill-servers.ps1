param(
  [int]$KeepPort = 0,     # e.g., 8580 to keep that server; 0 = kill all
  [switch]$DryRun         # show what would be killed, don’t kill
)

$ErrorActionPreference = 'SilentlyContinue'

Write-Host "Scanning for Streamlit/Python servers..." -ForegroundColor Cyan

# Map PIDs -> ports they listen on
$tcp = Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -ge 8000 }
$pidToPorts = @{}
foreach ($c in $tcp) {
  if (-not $pidToPorts.ContainsKey($c.OwningProcess)) { $pidToPorts[$c.OwningProcess] = @() }
  $pidToPorts[$c.OwningProcess] += $c.LocalPort
}

# Find python/streamlit processes (streamlit runs inside python.exe)
$procs = Get-CimInstance Win32_Process | Where-Object {
  $_.Name -match 'python\.exe|streamlit\.exe'
}

# Flag processes that look like Streamlit servers (cmdline contains "streamlit run")
$target = foreach ($p in $procs) {
  $ports = @()
  if ($pidToPorts.ContainsKey($p.ProcessId)) { $ports = $pidToPorts[$p.ProcessId] | Sort-Object -Unique }
  $isStreamlit = ($p.CommandLine -match 'streamlit\s+run') -or ($p.Name -match 'streamlit\.exe')
  if ($isStreamlit -or ($ports | Where-Object { $_ -ge 8500 -and $_ -le 8999 })) {
    [PSCustomObject]@{
      PID        = $p.ProcessId
      Name       = $p.Name
      Ports      = ($ports -join ',')
      Cmd        = ($p.CommandLine -replace '\s+', ' ').Trim()
      Keep       = ($KeepPort -ne 0 -and ($ports -contains $KeepPort))
    }
  }
}

if (-not $target -or $target.Count -eq 0) {
  Write-Host "No Streamlit/Python servers found." -ForegroundColor Yellow
  exit 0
}

Write-Host "`nFound candidate servers:" -ForegroundColor Green
$target | Sort-Object Keep -Descending | Format-Table -AutoSize PID,Name,Ports,Keep,Cmd

# Build kill list (anything not marked Keep)
$kill = @()
foreach ($t in $target) {
  if ($KeepPort -ne 0 -and $t.Keep) { continue }
  $kill += $t
}

if ($KeepPort -ne 0 -and -not ($target | Where-Object { $_.Keep })) {
  Write-Host "`n[Note] No process found listening on port $KeepPort to keep alive." -ForegroundColor Yellow
}

if ($kill.Count -eq 0) {
  Write-Host "`nNothing to terminate." -ForegroundColor Yellow
  exit 0
}

Write-Host "`nAbout to terminate these processes:" -ForegroundColor Magenta
$kill | Format-Table -AutoSize PID,Name,Ports

if ($DryRun) {
  Write-Host "`n(DryRun) Skipping termination." -ForegroundColor Yellow
  exit 0
}

foreach ($k in $kill) {
  try {
    Stop-Process -Id $k.PID -Force -ErrorAction Stop
    Write-Host "Killed PID $($k.PID) ($($k.Name)) [Ports: $($k.Ports)]" -ForegroundColor Red
  } catch {
    Write-Host "Failed to kill PID $($k.PID): $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

Write-Host "`nDone." -ForegroundColor Green
