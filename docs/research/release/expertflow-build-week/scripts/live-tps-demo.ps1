param(
    [ValidateSet('Demo','Judge')][string]$Mode = 'Demo',
    [string]$Model = $env:EXPERTFLOW_MODEL_PATH,
    [string]$Runtime = $env:EXPERTFLOW_LLAMA_CLI,
    [string]$Output,
    [int]$GeneratedTokens = 512,
    [switch]$SkipHashVerification
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Pairs = if ($Mode -eq 'Demo') { 1 } else { 3 }
$layers = '0,1,2,3,4,5,6,7,8,9,15,20'
$expectedModelSha = '089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba'
$expectedRuntimeSha = '5d68046dcd26e2fd018aaeaad5f99cdb7d88eca6fc10935925f1d660f7009407'
if (-not $Output) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $Output = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) "ExpertFlow\live-tps\$stamp"
}

function Stop-LiveDemo([string]$Message) {
    Write-Host ""
    Write-Host "LIVE TPS DEMO UNAVAILABLE" -ForegroundColor Red
    Write-Host $Message
    Write-Host "Fallback: uv run expertflow demo --replay"
    exit 10
}

if (-not $Model -or -not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    Stop-LiveDemo 'Q6 model not found. Pass -Model or set EXPERTFLOW_MODEL_PATH.'
}
if (-not $Runtime -or -not (Test-Path -LiteralPath $Runtime -PathType Leaf)) {
    Stop-LiveDemo 'Patched llama-cli not found. Pass -Runtime or set EXPERTFLOW_LLAMA_CLI.'
}
if ($GeneratedTokens -lt 32) { throw 'GeneratedTokens must be at least 32.' }

Write-Host ""
Write-Host "EXPERTFLOW LIVE TPS" -ForegroundColor Green
Write-Host "===================" -ForegroundColor DarkGreen
Write-Host "Mode:             $Mode"
Write-Host "Matched pairs:    $Pairs"
Write-Host "Generated tokens: $GeneratedTokens per run"
Write-Host "Model:            $((Resolve-Path $Model).Path)"
Write-Host "Runtime:          $((Resolve-Path $Runtime).Path)"
Write-Host "Placement:        [$layers]"
Write-Host "Evidence:         $Output"

if (-not $SkipHashVerification) {
    Write-Host "Verifying model and runtime identity..."
    $modelSha = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
    $runtimeSha = (Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($modelSha -ne $expectedModelSha) { Stop-LiveDemo "Model SHA-256 mismatch: $modelSha" }
    if ($runtimeSha -ne $expectedRuntimeSha) { Stop-LiveDemo "Runtime SHA-256 mismatch: $runtimeSha" }
    Write-Host "Identity:         verified" -ForegroundColor Green
} else {
    Write-Host "Identity:         hash verification skipped explicitly" -ForegroundColor DarkYellow
}

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$runner = Join-Path $PSScriptRoot 'run_q6_selected_static_pairs.ps1'
$analyzer = Join-Path $PSScriptRoot 'analyze_q6_selected_static.py'
$raw = Join-Path $Output 'raw-results.json'
$summaryPath = Join-Path $Output 'summary.json'
$csvPath = Join-Path $Output 'run-pairs.csv'

Push-Location $root
try {
    & $runner -Runtime $Runtime -Model $Model -Output $Output -Pairs $Pairs `
        -GeneratedTokens $GeneratedTokens -StaticLayers $layers `
        -BaselineCudaGraphs on -CudaGraphs on -StaticPrecompute
    if ($LASTEXITCODE -ne 0) { throw "Matched runner failed with exit code $LASTEXITCODE." }
    uv run python $analyzer --input $raw --output-json $summaryPath --output-csv $csvPath
    if ($LASTEXITCODE -ne 0) { throw "Analyzer failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}

$summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
$stock = [double]$summary.off.decode_tps_mean
$expertflow = [double]$summary.on.decode_tps_mean
$gain = [double]$summary.paired_improvement_pct_mean

Write-Host ""
Write-Host "LIVE RESULT" -ForegroundColor Green
Write-Host "-----------"
Write-Host ("Stock          {0,7:N2} TPS" -f $stock)
Write-Host ("ExpertFlow     {0,7:N2} TPS" -f $expertflow) -ForegroundColor Green
Write-Host ("Improvement    {0,7:+0.00;-0.00;0.00}%" -f $gain) -ForegroundColor Green
Write-Host ("Stock peak     {0,7:N3} MiB" -f [double]$summary.off.process_owned_peak_mib)
Write-Host ("ExpertFlow peak{0,7:N3} MiB" -f [double]$summary.on.process_owned_peak_mib)
Write-Host "Evidence       $summaryPath"
Write-Host ""
if ($Mode -eq 'Demo') {
    Write-Host 'ONE-PAIR REHEARSAL - useful for a live recording, not the authoritative benchmark.' -ForegroundColor DarkYellow
} else {
    Write-Host 'THREE-PAIR JUDGE CHECK - inspect variance and raw rows before interpreting.' -ForegroundColor DarkYellow
}
Write-Host 'AUTHORITATIVE TEN-PAIR RESULT: 28.13 TPS vs 22.967 stock (+22.48%).'
