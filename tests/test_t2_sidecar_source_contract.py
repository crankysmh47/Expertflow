from pathlib import Path
import subprocess


LLAMA = Path(r"C:\models\expertflow\worktrees\llama-p2-layer24-async-prefetch")
EXPERTFLOW = Path(r"C:\models\expertflow\worktrees\p2-layer24-async-prefetch")


def test_t2_is_explicit_disabled_by_default_and_layer24_only() -> None:
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    assert 'getenv("EXPERTFLOW_T2_TEMPORAL_SIDECAR")' in backend
    assert "EXPERTFLOW_T2_LOG" in backend
    assert "EXPERTFLOW_T2_RUN_ID" in backend
    assert "T2 requires temporal shadow and blocking layer 24" in backend
    assert "expertflow_cache_set_physical_slot_count(" in backend
    assert "EXPERTFLOW_CACHE_SIDECAR_SLOT_COUNT" in backend


def test_t2_uses_two_existing_cuda_descriptors_without_p1_dependency() -> None:
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    assert "sched->expertflow_t2_enabled" in backend
    assert "sched->expertflow_t2_cuda_descriptor_count" in backend
    assert "EXPERTFLOW_SIDECAR_SLOT_COUNT" in backend
    assert "expertflow_p2_cuda_create(0, descriptor_count" in backend
    t2_start = backend.index('getenv("EXPERTFLOW_T2_TEMPORAL_SIDECAR")')
    t2_end = backend.index('getenv("EXPERTFLOW_PREDICTOR_SHADOW")')
    t2_config = backend[t2_start:t2_end]
    assert "expertflow_predictor_shadow_enabled" not in t2_config


def test_t2_keeps_one_packed_operation_and_requires_decode_identity() -> None:
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    probe = (EXPERTFLOW / "native/router_probe/main.cpp").read_text(encoding="utf-8")
    llama_h = (LLAMA / "include/llama.h").read_text(encoding="utf-8")
    backend_h = (LLAMA / "ggml/include/ggml-backend.h").read_text(encoding="utf-8")
    assert "expertflow_cache_plan_selected_with_external(" in backend
    assert "llama_expertflow_temporal_set_decode_identity(" in probe
    assert "llama_expertflow_temporal_set_decode_identity" in llama_h
    assert "ggml_backend_sched_expertflow_temporal_set_decode_identity" in backend_h


def test_t2_filters_against_projected_post_admission_cache_state() -> None:
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    start = backend.index("static bool expertflow_t2_enqueue_pending")
    end = backend.index(
        "bool ggml_backend_sched_expertflow_temporal_observe_router", start
    )
    enqueue = backend[start:end]
    assert "expertflow_cache_select_projected_absent_candidate(" in enqueue
    assert "current_selected" in enqueue
    assert "for (const expertflow_cache_slot & slot : cache_state.slots)" not in enqueue


def test_t2_does_not_modify_cuda_kernels_or_define_a_new_operation() -> None:
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "b9445cd5",
            "--",
            "ggml/src/ggml-cuda",
        ],
        cwd=LLAMA,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert changed == ""
    backend = (LLAMA / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    assert "GGML_OP_EXPERTFLOW" not in backend
