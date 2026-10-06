$ErrorActionPreference = 'Stop'
$repositoryCandidate = Join-Path $PSScriptRoot '..\release\expertflow-build-week'
$release = if (Test-Path -LiteralPath $repositoryCandidate) {
    (Resolve-Path $repositoryCandidate).Path
} else {
    (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
py (Join-Path $release 'scripts\verify_release.py')
if ($LASTEXITCODE -ne 0) { throw 'Release verification failed. Restore the archive and rerun scripts\verify-release.ps1.' }
