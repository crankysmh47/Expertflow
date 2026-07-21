from pathlib import Path


def test_cli_pair_runner_freezes_fair_stock_contract() -> None:
    source = Path("scripts/run_q6_selected_static_pairs.ps1").read_text(encoding="utf-8")

    assert "0,1,15,20" in source
    assert "--cpu-moe" in source
    assert "--ignore-eos" in source
    assert "GeneratedTokens = 512" in source
    assert "if ($pair % 2 -eq 1)" in source
    assert "Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage'" in source
    assert "catch { $owned = 0L }" in source
    assert "$promptCount.Groups[1].Value" in source
    assert "$generatedCount.Groups[1].Value" in source
    assert "[Security.Cryptography.SHA256]::Create()" in source
    assert "valid = $summary.Success" in source
    assert "LLAMA_EXPERTFLOW_SPLIT_PROFILE" in source
    assert "[string]$StaticLayers = '0,1,15,20'" in source
    assert "[string]$BaselineLayers = ''" in source
    assert "[ValidateSet('on','off')][string]$CudaGraphs = 'off'" in source
    assert "[ValidateSet('on','off')][string]$BaselineCudaGraphs = 'off'" in source
    assert "$env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER = $StaticLayers" in source
    assert "$env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER = $BaselineLayers" in source
    assert "if ($runCudaGraphs -eq 'off')" in source
    assert "Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS" in source
    assert "[switch]$StaticPrecompute" in source
    assert "[switch]$BaselineStaticPrecompute" in source
    assert "$env:LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE = '1'" in source
    assert "Remove-Item Env:LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE" in source
    assert "cuda_graphs = $runCudaGraphs" in source
    assert "static_precompute =" in source
    assert "[RUN $pair/$Pairs]" in source
    assert "[RESULT $mode]" in source


def test_live_tps_demo_exposes_recording_and_judge_modes() -> None:
    source = Path("scripts/live-tps-demo.ps1").read_text(encoding="utf-8")

    assert "[ValidateSet('Demo','Judge')]" in source
    assert "GeneratedTokens = 512" in source
    assert "if ($Mode -eq 'Demo') { 1 } else { 3 }" in source
    assert "run_q6_selected_static_pairs.ps1" in source
    assert "analyze_q6_selected_static.py" in source
    assert "uv run expertflow demo --replay" in source
    assert "ONE-PAIR REHEARSAL" in source
    assert "AUTHORITATIVE TEN-PAIR RESULT" in source
