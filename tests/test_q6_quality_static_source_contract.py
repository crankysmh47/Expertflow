from __future__ import annotations

import subprocess

from source_contract_paths import LLAMA


UPSTREAM = "a7312ae94f801fc9c6786dc56e38df57b964f697"


def _source(path: str) -> str:
    return (LLAMA / path).read_text(encoding="utf-8")


def test_static_island_is_disabled_by_default_and_layer_scoped() -> None:
    context = _source("src/llama-context.cpp")
    backend = _source("ggml/src/ggml-backend.cpp")

    assert 'getenv("LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER")' in context
    assert 'getenv("LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER")' in backend
    assert "sched->expertflow_static_island_layer = -1" in backend
    assert "expertflow_static_island_assign" in context
    assert 'snprintf(prefix, sizeof(prefix), "blk.%d.ffn_"' in backend
    assert 'strstr(tensor->name + prefix_len, "_exps.")' in backend


def test_static_island_owns_complete_persistent_bundle() -> None:
    backend = _source("ggml/src/ggml-backend.cpp")

    assert "expertflow_static_island_shadows[EXPERTFLOW_STATIC_ISLAND_MAX_SHADOWS]" in backend
    assert "ggml_backend_alloc_ctx_tensors_from_buft" in backend
    assert "ggml_backend_tensor_set_async" in backend
    assert "ggml_backend_buffer_free(sched->expertflow_static_island_shadows[i].buffer)" in backend
    assert "ggml_free(sched->expertflow_static_island_shadows[i].ctx)" in backend
    assert "source->data == nullptr || ggml_nbytes(source) != ggml_nbytes(shadow)" in backend


def test_static_island_keeps_boundary_and_scale_dependency_safe() -> None:
    context = _source("src/llama-context.cpp")

    assert "GGML_OP_MUL_MAT_ID" in context
    assert 'strcmp(name, "ffn_moe_weighted") == 0 || strcmp(name, "ffn_moe_out") == 0' in context
    mul_case = context[context.index("case GGML_OP_MUL:") : context.index("case GGML_OP_GLU:")]
    assert "tensor->src[0]" in mul_case
    assert "tensor->src[1]" not in mul_case


def test_static_island_has_no_diagnostic_bypass_or_new_cuda_operation() -> None:
    context = _source("src/llama-context.cpp")
    backend = _source("ggml/src/ggml-backend.cpp")
    combined = context + backend

    assert "LLAMA_EXPERTFLOW_STATIC_ISLAND_NO_SHADOW" not in combined
    assert "LLAMA_EXPERTFLOW_STATIC_ISLAND_NODE" not in combined
    assert "LLAMA_EXPERTFLOW_STATIC_ISLAND_MM_ONLY" not in combined
    assert "GGML_OP_EXPERTFLOW" not in combined

    changed_cuda = subprocess.run(
        ["git", "diff", "--name-only", UPSTREAM, "--", "ggml/src/ggml-cuda"],
        cwd=LLAMA,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert changed_cuda == ""
