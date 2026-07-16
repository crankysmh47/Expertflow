from pathlib import Path


LLAMA = Path(r"C:\models\expertflow\worktrees\llama-p2-layer24-async-prefetch")
EXPERTFLOW = Path(r"C:\models\expertflow\worktrees\p2-layer24-async-prefetch")


def test_temporal_runtime_is_disabled_by_default_and_shadow_only() -> None:
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    assert 'getenv("EXPERTFLOW_TEMPORAL_SHADOW")' in backend
    assert "EXPERTFLOW_TEMPORAL_ARTIFACT" in backend
    assert "EXPERTFLOW_TEMPORAL_SHADOW_LOG" in backend
    assert "EXPERTFLOW_TEMPORAL_RUN_ID" in backend
    assert "temporal shadow requires live cache disabled" in backend
    temporal_block = backend[
        backend.index('getenv("EXPERTFLOW_TEMPORAL_SHADOW")'):
        backend.index('getenv("EXPERTFLOW_P2_MODE")')
    ]
    assert "prefetch" not in temporal_block.lower()
    assert "cuda" not in temporal_block.lower()
    assert "if (sched->expertflow_predictor_shadow_enabled) {" in backend
    assert (
        "sched->expertflow_predictor_shadow_enabled ||\n"
        "        sched->expertflow_temporal_shadow_enabled"
    ) in backend
    predictor_start = backend.index('getenv("EXPERTFLOW_PREDICTOR_SHADOW")')
    initialization = backend[
        predictor_start:
        backend.index("if (sched->expertflow_p2_mode", predictor_start)
    ]
    assert (
        "sched->expertflow_predictor_shadow_enabled ||\n"
        "        sched->expertflow_temporal_shadow_enabled"
    ) not in initialization


def test_router_probe_resets_and_observes_decode_layer24_only() -> None:
    probe = (EXPERTFLOW / "native/router_probe/main.cpp").read_text(encoding="utf-8")
    assert "llama_expertflow_temporal_reset(context)" in probe
    assert "llama_expertflow_temporal_observe_router(" in probe
    assert 'state.phase == "decode"' in probe
    assert "layer_id == 24" in probe


def test_public_temporal_api_is_explicit() -> None:
    llama_h = (LLAMA / "include/llama.h").read_text(encoding="utf-8")
    backend_h = (LLAMA / "ggml/include/ggml-backend.h").read_text(encoding="utf-8")
    assert "llama_expertflow_temporal_reset" in llama_h
    assert "llama_expertflow_temporal_observe_router" in llama_h
    assert "ggml_backend_sched_expertflow_temporal_reset" in backend_h
    assert "ggml_backend_sched_expertflow_temporal_observe_router" in backend_h
