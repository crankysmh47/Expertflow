$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $root
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is missing. Install it from https://docs.astral.sh/uv/ and rerun this script.' }
    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { throw 'Setup failed. Run: uv sync --frozen' }
    uv run expertflow demo --replay
    if ($LASTEXITCODE -ne 0) { throw 'Replay failed. Run: uv run expertflow demo --replay' }
    Write-Host "Offline dashboard: $root\release\expertflow-build-week\dashboard.html"
} finally { Pop-Location }
