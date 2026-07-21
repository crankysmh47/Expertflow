param(
    [Parameter(Mandatory=$true)][string]$Runtime,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Pairs = 10,
    [int]$GeneratedTokens = 512,
    [string]$StaticLayers = '0,1,15,20',
    [string]$BaselineLayers = '',
    [ValidateSet('on','off')][string]$CudaGraphs = 'off',
    [ValidateSet('on','off')][string]$BaselineCudaGraphs = 'off',
    [switch]$StaticPrecompute,
    [switch]$BaselineStaticPrecompute
)

$ErrorActionPreference = 'Stop'
$cudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin'
$env:PATH = "$cudaBin;$(Split-Path $Runtime -Parent);$env:PATH"
$rows = @()
New-Item -ItemType Directory -Force -Path $Output | Out-Null

function Get-GpuState {
    $fields = (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used,memory.total --format=csv,noheader,nounits) -split ','
    return [ordered]@{
        temperature_c = [int]$fields[0].Trim()
        sm_clock_mhz = [int]$fields[1].Trim()
        used_mib = [int]$fields[2].Trim()
        total_mib = [int]$fields[3].Trim()
    }
}

for ($pair = 1; $pair -le $Pairs; $pair++) {
    if ($pair % 2 -eq 1) { $modes = @('off', 'on') } else { $modes = @('on', 'off') }
    for ($order = 1; $order -le 2; $order++) {
        $mode = $modes[$order - 1]
        $runName = ('pair-{0:D2}-{1}-{2}' -f $pair, $order, $mode)
        $runDir = Join-Path $Output $runName
        New-Item -ItemType Directory -Path $runDir | Out-Null
        $stdout = Join-Path $runDir 'stdout.log'
        $stderr = Join-Path $runDir 'stderr.log'
        Remove-Item Env:LLAMA_EXPERTFLOW_SPLIT_PROFILE -ErrorAction SilentlyContinue
        Remove-Item Env:GGML_SCHED_DEBUG -ErrorAction SilentlyContinue
        $runCudaGraphs = $(if ($mode -eq 'on') { $CudaGraphs } else { $BaselineCudaGraphs })
        if ($runCudaGraphs -eq 'off') {
            $env:GGML_CUDA_DISABLE_GRAPHS = '1'
        } else {
            Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
        }
        if (($mode -eq 'on' -and $StaticPrecompute) -or ($mode -eq 'off' -and $BaselineStaticPrecompute)) {
            $env:LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE = '1'
        } else {
            Remove-Item Env:LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE -ErrorAction SilentlyContinue
        }
        if ($mode -eq 'on') {
            $env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER = $StaticLayers
        } elseif ($BaselineLayers) {
            $env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER = $BaselineLayers
        } else {
            Remove-Item Env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER -ErrorAction SilentlyContinue
        }
        $before = Get-GpuState
        $arguments = @(
            '-m', $Model, '-p', 'Caching.', '-n', "$GeneratedTokens", '-c', '2048',
            '-ngl', '99', '--threads', '12', '--seed', '42', '--temp', '0',
            '--ignore-eos', '--verbose', '--cpu-moe', '--no-display-prompt', '--single-turn'
        )
        $label = $(if ($mode -eq 'on') { 'EXPERTFLOW' } else { 'STOCK' })
        Write-Host ""
        Write-Host "[RUN $pair/$Pairs] $label - $GeneratedTokens decode tokens" -ForegroundColor $(if ($mode -eq 'on') { 'Green' } else { 'DarkYellow' })
        Write-Host "  CUDA graphs: $runCudaGraphs | static layers: $(if ($mode -eq 'on') { $StaticLayers } else { 'none' })"
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $proc = Start-Process -FilePath $Runtime -WorkingDirectory (Split-Path $Runtime -Parent) `
            -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
            -PassThru -WindowStyle Hidden
        $peakSystem = $before.used_mib
        $peakOwned = 0L
        $samples = 0
        while (-not $proc.HasExited) {
            $gpu = Get-GpuState
            if ($gpu.used_mib -gt $peakSystem) { $peakSystem = $gpu.used_mib }
            $owned = 0L
            try {
                $owned = [long]((Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
                    Where-Object { $_.Status -eq 0 -and $_.InstanceName -match "pid_$($proc.Id)(_|$)" } |
                    Measure-Object CookedValue -Sum).Sum
            } catch { $owned = 0L }
            if ($owned -gt $peakOwned) { $peakOwned = $owned }
            $samples++
            Start-Sleep -Milliseconds 200
            $proc.Refresh()
        }
        $proc.WaitForExit()
        $watch.Stop()
        $text = [string](Get-Content -Raw $stdout -ErrorAction SilentlyContinue)
        $detail = [string](Get-Content -Raw $stderr -ErrorAction SilentlyContinue)
        $summary = [regex]::Match($text, '\[ Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s \]')
        $promptCount = [regex]::Matches($detail, 'prompt eval time =.*?/\s+(\d+) tokens') | Select-Object -Last 1
        $generatedCount = [regex]::Matches($detail, 'eval time =.*?/\s+(\d+) tokens') | Select-Object -Last 1
        $response = [regex]::Match($text, '(?s)>\s*Caching\.\s*(.*?)\s*\[\s*Prompt:').Groups[1].Value
        $afterImmediate = Get-GpuState
        Start-Sleep -Seconds 1
        $afterSettled = Get-GpuState
        $row = [ordered]@{
            pair = $pair
            order = $order
            mode = $mode
            command = @($Runtime) + $arguments
            static_layers = $(if ($mode -eq 'on') { $StaticLayers } elseif ($BaselineLayers) { $BaselineLayers } else { $null })
            cuda_graphs = $runCudaGraphs
            static_precompute = [bool](($mode -eq 'on' -and $StaticPrecompute) -or ($mode -eq 'off' -and $BaselineStaticPrecompute))
            exit_code = $proc.ExitCode
            valid = $summary.Success
            prompt_tps = $(if ($summary.Success) { [double]$summary.Groups[1].Value } else { $null })
            decode_tps = $(if ($summary.Success) { [double]$summary.Groups[2].Value } else { $null })
            prompt_tokens = $(if ($promptCount) { [int]$promptCount.Groups[1].Value } else { $null })
            generated_tokens = $(if ($generatedCount) { [int]$generatedCount.Groups[1].Value } else { $null })
            wall_seconds = [math]::Round($watch.Elapsed.TotalSeconds, 6)
            gpu_before = $before
            gpu_system_peak_mib = $peakSystem
            process_owned_peak_bytes = $peakOwned
            process_owned_peak_mib = [math]::Round($peakOwned / 1MB, 3)
            gpu_after_immediate = $afterImmediate
            gpu_after_settled = $afterSettled
            sample_count = $samples
            response_sha256 = $(if ($response) {
                $bytes = [Text.Encoding]::UTF8.GetBytes($response)
                $hasher = [Security.Cryptography.SHA256]::Create()
                try { ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
                finally { $hasher.Dispose() }
            } else { $null })
            stdout = $stdout
            stderr = $stderr
            stdout_sha256 = (Get-FileHash $stdout -Algorithm SHA256).Hash.ToLowerInvariant()
            stderr_sha256 = (Get-FileHash $stderr -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        $row | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runDir 'measurement.json') -Encoding utf8
        $rows += $row
        if ($row.valid) {
            Write-Host ("[RESULT $mode] decode {0:N2} TPS | prompt {1:N2} TPS | wall {2:N2}s | peak {3:N3} MiB" -f $row.decode_tps, $row.prompt_tps, $row.wall_seconds, $row.process_owned_peak_mib) -ForegroundColor $(if ($mode -eq 'on') { 'Green' } else { 'DarkYellow' })
        } else {
            Write-Host "[RESULT $mode] FAILED - inspect $stderr" -ForegroundColor Red
        }
    }
}

[ordered]@{
    schema_version = '1.0.0'
    measurement_kind = 'matched_cold_process_cli_pairs'
    runtime = (Resolve-Path $Runtime).Path
    model = (Resolve-Path $Model).Path
    generated_tokens = $GeneratedTokens
    pairs = $Pairs
    rows = $rows
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $Output 'raw-results.json') -Encoding utf8
