param(
    [Parameter(Mandatory=$true)][string]$Runtime,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Output,
    [int[]]$Contexts = @(2048, 8192, 16384, 32768),
    [int[]]$GpuLayers = @(99, 20, 10),
    [int]$GeneratedTokens = 8
)

$ErrorActionPreference = 'Stop'
$cudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin'
$env:PATH = "$cudaBin;$env:PATH"
$runRoot = Join-Path (Split-Path $Output -Parent) 'frontier-raw'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$rows = @()

foreach ($context in $Contexts) {
    foreach ($ngl in $GpuLayers) {
        $name = "ctx-$context-ngl-$ngl"
        $stdout = Join-Path $runRoot "$name.stdout.log"
        $stderr = Join-Path $runRoot "$name.stderr.log"
        $baseline = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim())
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $proc = Start-Process -FilePath $Runtime -WorkingDirectory (Split-Path $Runtime -Parent) -ArgumentList @(
            '-m', $Model, '-p', 'Caching.', '-n', "$GeneratedTokens",
            '-c', "$context", '-ngl', "$ngl", '--threads', '12', '--seed', '42', '--temp', '0',
            '--no-display-prompt', '--single-turn'
        ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -NoNewWindow
        $peakSystem = $baseline
        $peakProcessBytes = 0L
        while (-not $proc.HasExited) {
            $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim())
            if ($used -gt $peakSystem) { $peakSystem = $used }
            try {
                $samples = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples |
                    Where-Object { $_.InstanceName -match "pid_$($proc.Id)(_|$)" }
                $owned = [long](($samples | Measure-Object CookedValue -Sum).Sum)
                if ($owned -gt $peakProcessBytes) { $peakProcessBytes = $owned }
            } catch { }
            Start-Sleep -Milliseconds 200
            $proc.Refresh()
        }
        $proc.WaitForExit()
        $watch.Stop()
        $text = [string](Get-Content -Raw $stdout -ErrorAction SilentlyContinue)
        $timing = [regex]::Match($text, '\[ Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s \]')
        $rows += [ordered]@{
            context = $context; ngl = $ngl; exit_code = $proc.ExitCode
            stable = $timing.Success
            prompt_tps = $(if ($timing.Success) {[double]$timing.Groups[1].Value} else {$null})
            decode_tps = $(if ($timing.Success) {[double]$timing.Groups[2].Value} else {$null})
            wall_seconds = [math]::Round($watch.Elapsed.TotalSeconds, 3)
            process_owned_peak_bytes = $peakProcessBytes
            process_owned_peak_mib = [math]::Round($peakProcessBytes / 1MB, 3)
            gpu_system_baseline_mib = $baseline
            gpu_system_peak_mib = $peakSystem
            gpu_system_delta_peak_mib = $peakSystem - $baseline
            stdout = $stdout; stderr = $stderr
        }
    }
}

$result = [ordered]@{
    schema_version = '1.0.0'
    measurement_kind = 'measured_process_gpu_counter_with_nvidia_system_crosscheck'
    runtime = (Resolve-Path $Runtime).Path
    model = (Resolve-Path $Model).Path
    contexts = $Contexts
    gpu_layers = $GpuLayers
    generated_tokens = $GeneratedTokens
    safety_margin_mib = 256
    rows = $rows
}
$result | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $Output
Get-Content -Raw $Output
