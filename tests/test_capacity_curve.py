import pytest

from expertflow.analysis.capacity_curve import (
    build_capacity_curve,
    build_held_out_capacity_curve,
)
from expertflow.trace.schema import RouterTraceEvent


def event(
    token_index: int, layer_id: int, experts: tuple[int, ...]
) -> RouterTraceEvent:
    return RouterTraceEvent(
        schema_version="1.0.0",
        request_id="req-1",
        conversation_id="conv-1",
        turn_index=0,
        phase="prefill",
        forward_id=token_index,
        hook_order=token_index * 2 + layer_id,
        token_index=token_index,
        token_id=100 + token_index,
        layer_id=layer_id,
        selected_expert_ids=experts,
        selected_expert_weights=None,
        observed_at_ns=(token_index * 2 + layer_id) * 1_000,
    )


def test_builds_in_sample_capacity_and_transfer_curve() -> None:
    events = [
        event(token, layer, experts)
        for token, experts in enumerate(((1, 2), (1, 3), (1, 2)))
        for layer in (0, 1)
    ]

    report = build_capacity_curve(
        events,
        capacities=(2, 3),
        slot_bytes=100,
        expert_transfer_ms=0.5,
    )

    assert report["measurement_kind"] == "estimated"
    assert report["fit_scope"] == "in_sample"
    assert report["event_count"] == 6
    assert report["expert_demand_count"] == 12
    assert report["layer_ids"] == [0, 1]
    assert report["router_top_k"] == 2
    first = report["points"][0]
    assert first["capacity_per_layer"] == 2
    assert first["projected_cache_bytes"] == 400
    assert first["static_hotset"]["hit_rate"] == pytest.approx(5 / 6)
    assert first["static_hotset"][
        "estimated_serial_h2d_ms_per_layer_sweep"
    ] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("capacities", "slot_bytes", "transfer_ms"),
    [((), 1, 1.0), ((2,), 0, 1.0), ((2,), 1, 0.0)],
)
def test_rejects_invalid_capacity_curve_contract(
    capacities: tuple[int, ...], slot_bytes: int, transfer_ms: float
) -> None:
    with pytest.raises(ValueError):
        build_capacity_curve(
            [event(0, 0, (1, 2))],
            capacities=capacities,
            slot_bytes=slot_bytes,
            expert_transfer_ms=transfer_ms,
        )


def test_freezes_static_hotset_before_held_out_evaluation() -> None:
    training = [
        event(0, 0, (1, 2)),
        event(1, 0, (1, 2)),
        event(2, 0, (1, 3)),
    ]
    evaluation = [event(0, 0, (3, 4)), event(1, 0, (3, 4))]

    report = build_held_out_capacity_curve(
        training,
        evaluation,
        capacities=(2,),
        slot_bytes=100,
        expert_transfer_ms=0.5,
    )

    assert report["fit_scope"] == "held_out"
    assert report["training_event_count"] == 3
    assert report["evaluation_event_count"] == 2
    point = report["points"][0]
    assert point["static_hotset"]["hit_rate"] == 0.0
    assert point["lru"]["hit_rate"] == 0.5
    assert point["projected_cache_bytes"] == 200
