"""Causal replay views over the shared cache-policy engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from expertflow.analysis.cache_sim import PolicyName, policy_outcomes
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
) -> PolicyReplay:
    """Build a timeline from the same outcomes used by aggregate simulation."""

    outcomes = policy_outcomes(
        events, policy=policy, capacity_per_layer=capacity_per_layer
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
