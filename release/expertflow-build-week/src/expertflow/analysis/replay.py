"""Causal replay views over the shared cache-policy engine."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from expertflow.analysis.cache_sim import (
    PolicyEventOutcome,
    PolicyName,
    policy_outcomes,
)
from expertflow.trace.schema import RouterTraceEvent


@dataclass(frozen=True, slots=True)
class ReplayTimelineEvent:
    """One token/layer demand with ready and blocking experts."""

    request_id: str
    phase: str
    forward_id: int
    token_index: int
    token_id: int
    layer_id: int
    ready_expert_ids: tuple[int, ...]
    blocking_expert_ids: tuple[int, ...]
    status: Literal["ready", "blocking"]


@dataclass(frozen=True, slots=True)
class PolicyReplay:
    """Estimated causal replay for one cache policy."""

    measurement_kind: Literal["estimated"]
    policy: PolicyName
    capacity_per_layer: int
    event_count: int
    demand_count: int
    hit_count: int
    miss_count: int
    timeline: tuple[ReplayTimelineEvent, ...]


def replay_policy(
    events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    *,
    policy: PolicyName,
    capacity_per_layer: int,
    static_training_events: (
        list[RouterTraceEvent] | tuple[RouterTraceEvent, ...] | None
    ) = None,
) -> PolicyReplay:
    """Build a timeline from the same outcomes used by aggregate simulation."""

    if static_training_events is None:
        outcomes = policy_outcomes(
            events, policy=policy, capacity_per_layer=capacity_per_layer
        )
    else:
        if policy != "static_hotset":
            raise ValueError(
                "static_training_events require the static_hotset policy"
            )
        training = tuple(static_training_events)
        if not training:
            raise ValueError("static_training_events must not be empty")
        if capacity_per_layer <= 0 or any(
            len(event.selected_expert_ids) > capacity_per_layer
            for event in (*training, *events)
        ):
            raise ValueError("capacity cannot be below router top-k")
        counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        for event in training:
            counts[event.layer_id].update(event.selected_expert_ids)
        residents = {
            layer_id: frozenset(
                expert_id
                for expert_id, _ in sorted(
                    layer_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:capacity_per_layer]
            )
            for layer_id, layer_counts in counts.items()
        }
        missing_layers = {
            event.layer_id for event in events
        } - residents.keys()
        if missing_layers:
            raise ValueError(
                "training events do not cover every evaluation layer"
            )
        outcomes = tuple(
            PolicyEventOutcome(
                event,
                tuple(
                    expert
                    for expert in event.selected_expert_ids
                    if expert in residents[event.layer_id]
                ),
                tuple(
                    expert
                    for expert in event.selected_expert_ids
                    if expert not in residents[event.layer_id]
                ),
            )
            for event in events
        )
    timeline = tuple(
        ReplayTimelineEvent(
            request_id=outcome.event.request_id,
            phase=outcome.event.phase,
            forward_id=outcome.event.forward_id,
            token_index=outcome.event.token_index,
            token_id=outcome.event.token_id,
            layer_id=outcome.event.layer_id,
            ready_expert_ids=outcome.ready_expert_ids,
            blocking_expert_ids=outcome.blocking_expert_ids,
            status="blocking" if outcome.blocking_expert_ids else "ready",
        )
        for outcome in outcomes
    )
    hit_count = sum(len(event.ready_expert_ids) for event in timeline)
    miss_count = sum(len(event.blocking_expert_ids) for event in timeline)
    return PolicyReplay(
        measurement_kind="estimated",
        policy=policy,
        capacity_per_layer=capacity_per_layer,
        event_count=len(timeline),
        demand_count=hit_count + miss_count,
        hit_count=hit_count,
        miss_count=miss_count,
        timeline=timeline,
    )
