from __future__ import annotations

from source_contract_paths import LLAMA


def test_q1b_nll_sidecar_is_measurement_only() -> None:
    source = (LLAMA / "tools/perplexity/perplexity.cpp").read_text(encoding="utf-8")

    assert 'getenv("LLAMA_EXPERTFLOW_NLL_FILE")' in source
    assert '"token_index"' in source
    assert '"chunk_index"' in source
    assert '"token_id"' in source
    assert '"nll"' in source
    assert "token_nll_history[i] = v" in source


def test_q1b_nll_sidecar_does_not_change_island_sources() -> None:
    context = (LLAMA / "src/llama-context.cpp").read_text(encoding="utf-8")
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")

    assert "LLAMA_EXPERTFLOW_NLL_FILE" not in context
    assert "LLAMA_EXPERTFLOW_NLL_FILE" not in backend
