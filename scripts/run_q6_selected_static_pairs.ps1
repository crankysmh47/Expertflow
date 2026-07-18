param(
    [Parameter(Mandatory=$true)][string]$Runtime,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Pairs = 10,
    [int]$GeneratedTokens = 512
)

$ErrorActionPreference = 'Stop'
$cudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin'
$env:PATH = "$cudaBin;$(Split-Path $Runtime -Parent);$env:PATH"
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
$staticLayers = '0,1,15,20'
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
        if ($mode -eq 'on') {
            $env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER = $staticLayers
        } else {
            Remove-Item Env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER -ErrorAction SilentlyContinue
        }
        $before = Get-GpuState
        $arguments = @(
            '-m', $Model, '-p', 'Caching.', '-n', "$GeneratedTokens", '-c', '2048',
            '-ngl', '99', '--threads', '12', '--seed', '42', '--temp', '0',
            '--ignore-eos', '--verbose', '--cpu-moe', '--no-display-prompt', '--single-turn'
        )
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
            static_layers = $(if ($mode -eq 'on') { $staticLayers } else { $null })
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
