param([string]$Interpreter)
$ErrorActionPreference = 'Stop'
$repositoryCandidate = Join-Path $PSScriptRoot '../docs/research/release/expertflow-build-week'
$release = if (Test-Path -LiteralPath $repositoryCandidate) {
    (Resolve-Path $repositoryCandidate).Path
} else {
    (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
if (-not $Interpreter) {
    $launchers = if ($env:OS -eq 'Windows_NT') { @('py', 'python', 'python3') } else { @('python3', 'python', 'py') }
    foreach ($launcher in $launchers) {
        $command = Get-Command $launcher -ErrorAction SilentlyContinue
        if ($command) { $Interpreter = $command.Source; break }
    }
}
if (-not $Interpreter) { throw 'Python 3.11+ is required; pass -Interpreter with its executable path.' }
& $Interpreter (Join-Path $release 'scripts/verify_release.py')
if ($LASTEXITCODE -ne 0) { throw 'Release verification failed. Restore the archive and rerun scripts/verify-release.ps1.' }
