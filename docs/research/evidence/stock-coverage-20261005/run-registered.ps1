# One registered sequence only. The collector rejects existing output roots.
$ErrorActionPreference='Stop'
$widerRepo='C:\sem4\expertflow'
$widerWorkspace=Join-Path $widerRepo '.superpowers\sdd\2026-10-05-wider-stock-collector'
Set-Location -LiteralPath $widerRepo
$widerJob=[ordered]@{status='RUNNING';pid=$PID;started_at_utc=[DateTime]::UtcNow.ToString('o');command='uv run --no-sync expertflow stock coverage run';exit_code=$null}
$widerJob | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $widerWorkspace 'native-job.json') -Encoding utf8
try {
    & uv run --no-sync expertflow stock coverage run *> (Join-Path $widerWorkspace 'native-collection.log')
    $widerJob.exit_code=$LASTEXITCODE
    $widerJob.status='COMPLETED'
} catch {
    $widerJob.status='SUPERVISOR-FAILED'
    $widerJob.reason=$_.Exception.Message
} finally {
    $widerJob.finished_at_utc=[DateTime]::UtcNow.ToString('o')
    $widerJob | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $widerWorkspace 'native-job.json') -Encoding utf8
}
