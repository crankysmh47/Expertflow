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
    assert "$env:GGML_CUDA_DISABLE_GRAPHS = '1'" in source
