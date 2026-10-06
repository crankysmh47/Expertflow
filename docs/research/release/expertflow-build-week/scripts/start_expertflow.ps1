param([string]$Deployment = "deployments/max-agentic.json", [int]$Port = 8080)
$ErrorActionPreference = "Stop"
if (-not $env:EXPERTFLOW_MODEL_PATH) { throw "Set EXPERTFLOW_MODEL_PATH" }
if (-not $env:EXPERTFLOW_LLAMA_SERVER) { throw "Set EXPERTFLOW_LLAMA_SERVER" }
$root = Split-Path $PSScriptRoot -Parent
$profilePath = Join-Path $root $Deployment
$profile = Get-Content -Raw -LiteralPath $profilePath | ConvertFrom-Json
$pidFile = Join-Path $root ".expertflow-server.pid"
if (Test-Path $pidFile) { throw "PID file already exists: $pidFile" }
$out = Join-Path $root "expertflow-server.stdout.log"
$err = Join-Path $root "expertflow-server.stderr.log"
$env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER = ($profile.placement.static_expert_layers -join ',')
$env:LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE = '1'
$cudaBin = Join-Path ${env:CUDA_PATH} 'bin'
$env:PATH = "$cudaBin;$(Split-Path $env:EXPERTFLOW_LLAMA_SERVER -Parent);$env:PATH"
$arguments = @('-m', $env:EXPERTFLOW_MODEL_PATH, '--host', '127.0.0.1', '--port', "$Port", '-c', "$($profile.context)", '-np', "$($profile.parallel_slots)", '-ngl', '99', '--threads', '12', '--cpu-moe')
$process = Start-Process -FilePath $env:EXPERTFLOW_LLAMA_SERVER -ArgumentList $arguments -WorkingDirectory (Split-Path $env:EXPERTFLOW_LLAMA_SERVER -Parent) -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
$process.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
Write-Output "PID=$($process.Id) health=http://127.0.0.1:$Port/health base=http://127.0.0.1:$Port/v1"
