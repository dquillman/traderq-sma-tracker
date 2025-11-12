# PowerShell script to create a desktop shortcut with custom icon

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "TraderQ SMA Tracker.lnk"
$TargetPath = Join-Path $ScriptDir "START_TRADERQ.bat"
$IconPath = Join-Path $ScriptDir "traderq_icon.ico"

# Create WScript Shell object
$WScriptShell = New-Object -ComObject WScript.Shell

# Create the shortcut
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath
$Shortcut.WorkingDirectory = $ScriptDir
$Shortcut.Description = "Launch TraderQ SMA 20/200 Tracker"
$Shortcut.IconLocation = $IconPath

# Save the shortcut
$Shortcut.Save()

Write-Host "Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "Location: $ShortcutPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now double-click 'TraderQ SMA Tracker' on your desktop to launch the app!" -ForegroundColor Yellow
