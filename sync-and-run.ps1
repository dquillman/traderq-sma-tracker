# sync-and-run.ps1
param(
  [string]$Msg = "chore: bump & run",
  [int]$Port = 8581
)

$ErrorActionPreference = 'Stop'
cd G:\Users\daveq\traderq

.\bump-and-push.ps1 -Msg $Msg
.\start-traderq.ps1 -Port $Port
