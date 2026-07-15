import pytest

from expertflow.analysis.deadline import evaluate_one_layer_oracle
from expertflow.trace.schema import RouterTraceEvent


def event(
    forward_id: int,
    layer_id: int,
    experts: tuple[int, ...],
    observed_at_ns: int,
    *,
    phase: str = "decode",
) -> RouterTraceEvent:
    return RouterTraceEvent(
        schema_version="1.0.0",
        request_id="req-1",
        conversation_id="conv-1",
        turn_index=0,
        phase=phase,
        forward_id=forward_id,
        hook_order=layer_id,
        token_index=forward_id,
        token_id=100 + forward_id,
        layer_id=layer_id,
        selected_expert_ids=experts,
        selected_expert_weights=None,
        observed_at_ns=observed_at_ns,
    )


def test_evaluates_frozen_static_cache_with_one_layer_oracle() -> None:
    training = [
        event(0, 0, (1, 2), 0, phase="prefill"),
        event(0, 1, (1, 2), 1, phase="prefill"),
    ]
    evaluation = [
        event(1, 0, (3, 4), 1_000_000),
        event(1, 1, (3, 4), 3_000_000),
    ]

    report = evaluate_one_layer_oracle(
        training,
        [evaluation],
        capacity_per_layer=2,
        expert_transfer_ms=0.5,
    )

    assert report["measurement_kind"] == "estimated"
    assert report["window_measurement_kind"] == "measured_backend_specific"
    assert report["fit_scope"] == "held_out"
    assert report["token_count"] == 1
    assert report["event_count"] == 2
    assert report["static_hit_rate"] == 0.0
    assert report["blocking_no_prefetch_ms_per_token"] == pytest.approx(2.0)
    oracle = report["one_layer_oracle"]
    assert oracle["on_time_event_count"] == 1
    assert oracle["late_event_count"] == 1
    assert oracle["residual_blocking_ms_per_token"] == pytest.approx(1.0)
    assert report["timeline"][1]["available_window_ms"] == pytest.approx(2.0)


def test_deadline_evaluation_rejects_non_monotonic_timestamps() -> None:
    training = [
        event(0, 0, (1, 2), 0, phase="prefill"),
        event(0, 1, (1, 2), 1, phase="prefill"),
    ]
    evaluation = [
        event(1, 0, (1, 2), 2_000_000),
        event(1, 1, (1, 2), 1_000_000),
    ]

    with pytest.raises(ValueError, match="monotonic"):
        evaluate_one_layer_oracle(
            training,
            [evaluation],
            capacity_per_layer=2,
            expert_transfer_ms=0.5,
        )


def test_labels_cross_backend_deadline_estimate_at_serialization_boundary() -> None:
    training = [
        event(0, 0, (1, 2), 0, phase="prefill"),
        event(0, 1, (1, 2), 1, phase="prefill"),
    ]
    evaluation = [
        event(1, 0, (3, 4), 1_000_000),
        event(1, 1, (3, 4), 3_000_000),
    ]

    with pytest.raises(ValueError, match="both be declared"):
        evaluate_one_layer_oracle(
            training,
            [evaluation],
            capacity_per_layer=2,
            expert_transfer_ms=0.5,
            transfer_backend="cuda",
        )

    report = evaluate_one_layer_oracle(
        training,
        [evaluation],
        capacity_per_layer=2,
        expert_transfer_ms=0.5,
        transfer_backend="cuda_idle_microbenchmark",
        window_backend="vulkan_router_callback",
        transfer_statistic="pooled_p95",
    )

    assert report["measurement_kind"] == "estimated_cross_backend"
    assert report["timing_evidence"] == {
        "transfer_backend": "cuda_idle_microbenchmark",
        "window_backend": "vulkan_router_callback",
        "transfer_statistic": "pooled_p95",
        "contention_measured": False,
        "live_runtime_measurement": False,
    }
