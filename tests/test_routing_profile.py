from dataclasses import replace

from expertflow.analysis.profile import summarize_routing
from expertflow.trace.schema import RouterTraceEvent


def event(token_index: int, experts: tuple[int, ...]) -> RouterTraceEvent:
    return RouterTraceEvent(
        schema_version="1.0.0",
        request_id="req-1",
        conversation_id="conv-1",
        turn_index=0,
        phase="decode",
        forward_id=token_index,
        hook_order=token_index,
        token_index=token_index,
        token_id=100 + token_index,
        layer_id=4,
        selected_expert_ids=experts,
        selected_expert_weights=None,
        observed_at_ns=token_index * 1_000,
    )


def test_summarizes_layer_locality_and_static_hit_curve() -> None:
    profile = summarize_routing(
        [event(0, (1, 2)), event(1, (1, 3)), event(2, (1, 2))],
        static_budgets=(1, 2),
    )

    assert profile.total_events == 3
    assert len(profile.layers) == 1
    layer = profile.layers[0]
    assert layer.layer_id == 4
    assert layer.event_count == 3
    assert layer.selection_count == 6
    assert layer.unique_expert_count == 3
    assert layer.top_experts == ((1, 3), (2, 2), (3, 1))
    assert layer.static_hit_rates == ((1, 0.5), (2, 5 / 6))
    assert layer.adjacent_reuse_rate == 0.5
    assert layer.mean_reuse_distance_tokens == 4 / 3


def test_adjacent_reuse_resets_between_requests() -> None:
    first = event(0, (1, 2))
    second = replace(
        first,
        request_id="req-2",
        conversation_id="conv-2",
        hook_order=1,
    )

    layer = summarize_routing([first, second]).layers[0]

    assert layer.adjacent_reuse_rate is None
    assert layer.mean_reuse_distance_tokens is None


def test_empty_profile_is_valid() -> None:
    profile = summarize_routing([])

    assert profile.total_events == 0
    assert profile.layers == ()
