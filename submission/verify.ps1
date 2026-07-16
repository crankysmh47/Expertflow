$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$expectedReportHash =
    'F3DC647D9965D726771632421B8FA5DFFDDC165D3EBAE49F6F10381BBB75A90C'
$simulationPath = Join-Path $env:TEMP (
    'expertflow-submission-' + [guid]::NewGuid().ToString('N') + '.json'
)

Push-Location $root
try {
    $env:PYTHONPATH = '.'
    uv run pytest tests/test_replay_fixture.py -q
    if ($LASTEXITCODE -ne 0) {
        throw "Replay fixture test failed with exit code $LASTEXITCODE"
    }

    uv run expertflow simulate examples\replay\trace.jsonl `
        --capacity-per-layer 8 `
        --output $simulationPath
    if ($LASTEXITCODE -ne 0) {
        throw "Replay simulation failed with exit code $LASTEXITCODE"
    }

    $simulation = Get-Content -LiteralPath $simulationPath -Raw |
        ConvertFrom-Json
    $static = $simulation.simulation.static_hotset
    $lru = $simulation.simulation.lru
    if (
        $static.demand_count -ne 64 -or
        $static.hit_count -ne 26 -or
        $lru.hit_count -ne 19
    ) {
        throw "Replay totals do not match the checked-in fixture"
    }

    $report = Join-Path $PSScriptRoot 'observatory.html'
    $reportHash = (Get-FileHash -LiteralPath $report -Algorithm SHA256).Hash
    if ($reportHash -ne $expectedReportHash) {
        throw "Bundled Observatory hash does not match"
    }

    Write-Output 'ExpertFlow submission verification passed.'
    Write-Output 'events=8 demands=64 static_hits=26 lru_hits=19'
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $simulationPath) {
        Remove-Item -LiteralPath $simulationPath -Force
    }
}
