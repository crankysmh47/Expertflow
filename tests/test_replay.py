from expertflow.analysis.cache_sim import simulate_policies
from expertflow.analysis.replay import replay_policy
from expertflow.trace.schema import RouterTraceEvent


def event(token_index: int, experts: tuple[int, ...]) -> RouterTraceEvent:
    return RouterTraceEvent(
        schema_version="1.0.0",
        request_id="req-001",
        conversation_id="conv-001",
        turn_index=0,
        phase="decode",
        forward_id=token_index,
        hook_order=token_index,
        token_index=token_index,
        token_id=100 + token_index,
        layer_id=7,
        selected_expert_ids=experts,
        selected_expert_weights=None,
        observed_at_ns=1_000 + token_index,
    )


def test_static_replay_matches_aggregate_simulator() -> None:
    events = (
        event(0, (1, 2)),
        event(1, (1, 3)),
        event(2, (1, 2)),
    )

    replay = replay_policy(events, policy="static_hotset", capacity_per_layer=2)
    simulation = simulate_policies(events, capacity_per_layer=2)

    assert replay.measurement_kind == "estimated"
    assert replay.hit_count == simulation.static_hotset.hit_count
    assert replay.miss_count == simulation.static_hotset.miss_count
    assert replay.timeline[0].ready_expert_ids == (1, 2)
    assert replay.timeline[1].blocking_expert_ids == (3,)
    assert replay.timeline[1].status == "blocking"


def test_lru_replay_preserves_causal_identity() -> None:
    events = (event(0, (1, 2)), event(1, (1, 3)))

    replay = replay_policy(events, policy="lru", capacity_per_layer=2)

    second = replay.timeline[1]
    assert second.request_id == "req-001"
    assert second.forward_id == 1
    assert second.token_index == 1
    assert second.layer_id == 7
    assert second.ready_expert_ids == (1,)
    assert second.blocking_expert_ids == (3,)


def test_static_replay_can_use_a_frozen_training_hotset() -> None:
    training = (
        event(0, (1, 2)),
        event(1, (1, 2)),
        event(2, (1, 3)),
    )
    evaluation = (event(0, (3, 4)), event(1, (3, 4)))

    replay = replay_policy(
        evaluation,
        policy="static_hotset",
        capacity_per_layer=2,
        static_training_events=training,
    )

    assert replay.hit_count == 0
    assert replay.miss_count == 4
    assert replay.timeline[0].blocking_expert_ids == (3, 4)
