param(
    [Parameter(Mandatory=$true)][int]$ProcessId,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$IntervalMs = 200
)

$ErrorActionPreference = 'SilentlyContinue'
$samples = @()
$peakOwned = 0L
while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
    $gpu = (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader,nounits) -split ','
    $owned = [long]((Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples |
        Where-Object { $_.InstanceName -match "pid_${ProcessId}(_|$)" } |
        Measure-Object CookedValue -Sum).Sum
    if ($owned -gt $peakOwned) { $peakOwned = $owned }
    if ($gpu.Count -ge 3) {
        $samples += [ordered]@{
            timestamp = (Get-Date).ToUniversalTime().ToString('o')
            temperature_c = [int]$gpu[0].Trim()
            sm_clock_mhz = [int]$gpu[1].Trim()
            system_used_mib = [int]$gpu[2].Trim()
            process_owned_bytes = $owned
        }
    }
    Start-Sleep -Milliseconds $IntervalMs
}
[ordered]@{
    interval_ms = $IntervalMs
    peak_process_owned_bytes = $peakOwned
    peak_process_owned_mib = [math]::Round($peakOwned / 1MB, 3)
    samples = $samples
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Output -Encoding utf8
