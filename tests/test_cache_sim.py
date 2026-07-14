from expertflow.analysis.cache_sim import simulate_policies
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


def test_compares_reactive_static_and_lru_under_one_budget() -> None:
    events = [event(0, (1, 2)), event(1, (1, 3)), event(2, (1, 2))]

    report = simulate_policies(events, capacity_per_layer=2)

    assert report.measurement_kind == "estimated"
    assert report.capacity_per_layer == 2
    assert report.reactive.demand_count == 6
    assert report.reactive.hit_count == 0
    assert report.reactive.load_count == 6
    assert report.static_hotset.hit_count == 5
    assert report.static_hotset.miss_count == 1
    assert report.static_hotset.preload_count == 2
    assert report.lru.hit_count == 2
    assert report.lru.miss_count == 4
    assert report.lru.hit_rate == 1 / 3


def test_rejects_capacity_below_router_top_k() -> None:
    try:
        simulate_policies([event(0, (1, 2))], capacity_per_layer=1)
    except ValueError as error:
        assert "top-k" in str(error)
    else:
        raise AssertionError("impossible capacity was accepted")
