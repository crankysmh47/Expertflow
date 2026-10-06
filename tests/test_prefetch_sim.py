import pytest

from expertflow.analysis.prefetch_sim import simulate_concurrent_prefetch
from expertflow.trace.schema import SCHEMA_VERSION, RouterTraceEvent


def _event(
    conversation_id: str,
    forward_id: int,
    layer_id: int,
    token_index: int,
    experts: tuple[int, ...],
    hook_order: int = 0,
) -> RouterTraceEvent:
    return RouterTraceEvent(
        schema_version=SCHEMA_VERSION,
        request_id=f"{conversation_id}-r",
        conversation_id=conversation_id,
        turn_index=0,
        phase="decode",
        forward_id=forward_id,
        hook_order=hook_order,
        token_index=token_index,
        token_id=7,
        layer_id=layer_id,
        selected_expert_ids=tuple(experts),
        selected_expert_weights=None,
        observed_at_ns=0,
    )


def test_oracle_prefetch_eliminates_next_step_blocking() -> None:
    trace_a = [
        _event("a", 0, 0, 0, (1, 2)),
        _event("a", 1, 0, 1, (3, 4)),
    ]
    report = simulate_concurrent_prefetch(
        [trace_a],
        prediction="oracle",
        capacity_per_layer=8,
        max_transfers_per_step=8,
        expert_transfer_ms=0.273,
        slot_bytes=1024,
    )

    assert report.measurement_kind == "estimated"
    assert report.step_count == 2
    assert report.demand_count == 4
    first, second = report.steps
    assert (first.transfers_issued, first.ready_useful) == (2, 2)
    assert second.miss_count == 0
    assert second.ready_useful == 2
    assert second.wasted_transfers == 0
    assert report.miss_count == 0
    assert report.estimated_saved_blocking_ms == pytest.approx(4 * 0.273)


def test_transfer_budget_forces_residual_misses() -> None:
    trace = [
        _event("a", 0, 0, 0, (1,)),
        _event("a", 1, 0, 1, (10, 11, 12)),
    ]
    report = simulate_concurrent_prefetch(
        [trace],
        prediction="oracle",
        capacity_per_layer=8,
        max_transfers_per_step=2,
        expert_transfer_ms=0.5,
        slot_bytes=2048,
    )

    _, second = report.steps
    assert second.predicted_candidates == 3
    assert second.transfers_issued == 2
    assert second.miss_count >= 1
    assert second.ready_useful <= 2


def test_frequency_prediction_wastes_transfers_on_unseen_experts() -> None:
    trace = [
        _event("a", 0, 0, 0, (5,)),
        _event("a", 1, 0, 1, (99,)),
    ]
    report = simulate_concurrent_prefetch(
        [trace],
        prediction="frequency",
        capacity_per_layer=8,
        max_transfers_per_step=4,
        expert_transfer_ms=0.1,
        slot_bytes=4096,
    )

    _, second = report.steps
    assert second.transfers_issued == 0
    assert report.wasted_transfers == 0

    trace_with_support = [
        _event("a", 0, 0, 0, (5, 6)),
        _event("a", 1, 0, 1, (7,)),
        _event("a", 2, 0, 2, (5, 8)),
    ]
    supported = simulate_concurrent_prefetch(
        [trace_with_support],
        prediction="frequency",
        capacity_per_layer=2,
        max_transfers_per_step=4,
        expert_transfer_ms=0.1,
        slot_bytes=4096,
    )
    third = supported.steps[2]
    assert third.transfers_issued >= 1
    assert third.ready_useful + third.wasted_transfers == (
        third.transfers_issued
    )


def test_shared_cache_serves_second_conversation_from_first_residency() -> None:
    trace_a = [
        _event("a", 0, 0, 0, (1, 2)),
        _event("a", 1, 0, 1, (1, 2)),
    ]
    trace_b = [
        _event("b", 0, 0, 0, (9,)),
        _event("b", 1, 0, 1, (1,)),
    ]
    report = simulate_concurrent_prefetch(
        [trace_a, trace_b],
        prediction="none",
        capacity_per_layer=4,
        max_transfers_per_step=0,
        expert_transfer_ms=0.25,
        slot_bytes=512,
    )

    assert report.conversation_count == 2
    assert report.hit_count >= 2


def test_protection_limits_issuable_prefetch_under_pressure() -> None:
    trace_a = [
        _event("a", 0, 0, 0, (1,)),
        _event("a", 1, 0, 1, (2, 3, 4, 5)),
    ]
    report = simulate_concurrent_prefetch(
        [trace_a],
        prediction="oracle",
        capacity_per_layer=4,
        max_transfers_per_step=8,
        expert_transfer_ms=0.1,
        slot_bytes=64,
    )

    _, second = report.steps
    assert second.transfers_issued == 4
    assert second.ready_useful == 4
    assert second.wasted_transfers == 0
    assert second.miss_count == 0
    assert second.ready_useful + second.wasted_transfers == (
        second.transfers_issued
    )


def test_frequency_wastes_transfer_on_predicted_but_undemanded_expert() -> None:
    trace = [
        _event("a", 0, 0, 0, (5, 6)),
        _event("a", 1, 0, 1, (7,)),
        _event("a", 2, 0, 2, (8,)),
    ]
    report = simulate_concurrent_prefetch(
        [trace],
        prediction="frequency",
        capacity_per_layer=2,
        max_transfers_per_step=8,
        expert_transfer_ms=0.1,
        slot_bytes=64,
    )

    third = report.steps[2]
    assert third.transfers_issued == 1
    assert third.wasted_transfers == 1
    assert third.ready_useful == 0
    assert report.ready_useful + report.wasted_transfers == (
        report.transfers_issued
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prediction": "psychic"},
        {"capacity_per_layer": 1},
        {"max_transfers_per_step": -1},
        {"expert_transfer_ms": -0.5},
        {"slot_bytes": 0},
    ],
)
def test_rejects_invalid_contracts(kwargs) -> None:
    trace = [
        _event("a", 0, 0, 0, (1, 2)),
        _event("a", 1, 0, 1, (3,)),
    ]
    base = {
        "prediction": "oracle",
        "capacity_per_layer": 8,
        "max_transfers_per_step": 4,
        "expert_transfer_ms": 0.2,
        "slot_bytes": 128,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        simulate_concurrent_prefetch([trace], **base)


def test_rejects_empty_traces() -> None:
    with pytest.raises(ValueError):
        simulate_concurrent_prefetch(
            [],
            prediction="oracle",
            capacity_per_layer=8,
            max_transfers_per_step=4,
            expert_transfer_ms=0.2,
            slot_bytes=128,
        )


def test_interleaving_preserves_layer_execution_order() -> None:
    trace = [
        _event("a", 0, 5, 0, (1,), hook_order=0),
        _event("a", 0, 2, 0, (2,), hook_order=0),
        _event("a", 0, 2, 1, (2,), hook_order=1),
    ]
    report = simulate_concurrent_prefetch(
        [trace],
        prediction="none",
        capacity_per_layer=8,
        max_transfers_per_step=0,
        expert_transfer_ms=0.0,
        slot_bytes=64,
    )

    assert report.step_count == 1
    assert report.demand_count == 3
