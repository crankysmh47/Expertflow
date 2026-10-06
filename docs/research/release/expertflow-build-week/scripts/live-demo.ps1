param([string]$Model = $env:EXPERTFLOW_MODEL_PATH, [string]$Runtime = $env:EXPERTFLOW_LLAMA_CLI, [string]$Server = $env:EXPERTFLOW_LLAMA_SERVER)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Model) { throw 'Model path missing. Set EXPERTFLOW_MODEL_PATH or pass -Model <path>.' }
if (-not $Runtime) { throw 'Runtime path missing. Set EXPERTFLOW_LLAMA_CLI or pass -Runtime <path>.' }
if (-not $Server) { throw 'Server path missing. Set EXPERTFLOW_LLAMA_SERVER or pass -Server <path>.' }
Push-Location $root
try {
    uv run expertflow doctor --model "$Model" --runtime "$Runtime" --server "$Server"
    if ($LASTEXITCODE -ne 0) { throw 'Doctor failed. Correct the reported prerequisite, then rerun this script.' }
    uv run expertflow run deployments/max-performance.json --model "$Model" --runtime "$Runtime"
} finally { Pop-Location }
