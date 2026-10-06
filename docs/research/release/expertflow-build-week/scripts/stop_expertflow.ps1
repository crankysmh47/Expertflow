$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$pidFile = Join-Path $root ".expertflow-server.pid"
if (-not (Test-Path $pidFile)) { Write-Output "ExpertFlow is not running"; exit 0 }
$serverPid = [int](Get-Content -Raw -LiteralPath $pidFile)
$process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if ($process) { Stop-Process -Id $serverPid; $process.WaitForExit(30000) }
Remove-Item -LiteralPath $pidFile -Force
Write-Output "ExpertFlow stopped"
