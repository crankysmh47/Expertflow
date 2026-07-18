from __future__ import annotations

from source_contract_paths import LLAMA


def test_q1b_static_island_list_is_bounded_and_backward_compatible():
    context = (LLAMA / "src/llama-context.cpp").read_text(encoding="utf-8")
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")

    assert "EXPERTFLOW_STATIC_ISLAND_MAX_LAYERS 4" in backend
    assert "EXPERTFLOW_STATIC_ISLAND_MAX_SHADOWS 16" in backend
    assert "expertflow_static_island_layers[EXPERTFLOW_STATIC_ISLAND_MAX_LAYERS]" in backend
    assert "expertflow_static_island_has_layer" in context
    assert "expertflow_static_island_layer = -1" in backend
    assert "LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER" in context
    assert "LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER" in backend


def test_q1b_static_island_list_rejects_overflow_and_duplicates():
    context = (LLAMA / "src/llama-context.cpp").read_text(encoding="utf-8")
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    combined = context + backend

    assert "too many LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER entries" in combined
    assert "duplicate LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER entry" in combined
