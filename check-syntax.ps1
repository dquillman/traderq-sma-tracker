# check-syntax.ps1  -- plain ASCII, robust
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Set-Location G:\Users\daveq\traderq

# pick python from venv if present
$python = "G:\Users\daveq\traderq\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$errors = @()

Get-ChildItem -Path . -Filter *.py -Recurse | ForEach-Object {
    $file = $_.FullName
    # compile file; capture stderr to a temp file
    $tmp = [System.IO.Path]::GetTempFileName()
    $p = Start-Process -FilePath $python `
        -ArgumentList @("-m","py_compile",$file) `
        -PassThru -NoNewWindow -RedirectStandardError $tmp -WindowStyle Hidden
    $p.WaitForExit()
    $err = ""
    if (Test-Path $tmp) { $err = (Get-Content $tmp -Raw) }
    Remove-Item $tmp -ErrorAction SilentlyContinue

    if ($p.ExitCode -ne 0 -or $err) {
        Write-Host "SYNTAX ERROR: $file"
        if ($err) { Write-Host $err.Trim() }
        $errors += $file
    }
}

if ($errors.Count -eq 0) {
    Write-Host ""
    Write-Host "OK: All Python files compiled cleanly."
    exit 0
} else {
    Write-Host ""
    Write-Host ("Found {0} file(s) with syntax errors." -f $errors.Count)
    exit 1
}
