"""Slot-budget cache-policy simulation over measured router demand."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Literal

from expertflow.trace.schema import RouterTraceEvent


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Estimated demand outcome for one cache policy."""

    policy: str
    demand_count: int
    hit_count: int
    miss_count: int
    load_count: int
    preload_count: int
    hit_rate: float


@dataclass(frozen=True, slots=True)
class CacheSimulationReport:
    """Comparable policy results under one per-layer slot budget."""

    measurement_kind: Literal["estimated"]
    capacity_per_layer: int
    reactive: PolicyResult
    static_hotset: PolicyResult
    lru: PolicyResult


PolicyName = Literal["reactive", "static_hotset", "lru"]


@dataclass(frozen=True, slots=True)
class PolicyEventOutcome:
    """One causal demand outcome shared by aggregate and replay views."""

    event: RouterTraceEvent
    ready_expert_ids: tuple[int, ...]
    blocking_expert_ids: tuple[int, ...]


def _result(
    policy: str,
    *,
    demand_count: int,
    hit_count: int,
    load_count: int,
    preload_count: int = 0,
) -> PolicyResult:
    miss_count = demand_count - hit_count
    return PolicyResult(
        policy=policy,
        demand_count=demand_count,
        hit_count=hit_count,
        miss_count=miss_count,
        load_count=load_count,
        preload_count=preload_count,
        hit_rate=hit_count / demand_count if demand_count else 0.0,
    )


def _validated_trace(
    events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    capacity_per_layer: int,
) -> tuple[RouterTraceEvent, ...]:
    if capacity_per_layer <= 0:
        raise ValueError("capacity_per_layer must be positive")
    trace = tuple(events)
    if any(
        len(event.selected_expert_ids) > capacity_per_layer for event in trace
    ):
        raise ValueError("capacity_per_layer cannot be below router top-k")
    return trace


def _static_residents(
    trace: tuple[RouterTraceEvent, ...], capacity_per_layer: int
) -> dict[int, frozenset[int]]:
    layer_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for event in trace:
        layer_counts[event.layer_id].update(event.selected_expert_ids)
    return {
        layer_id: frozenset(
            expert_id
            for expert_id, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:capacity_per_layer]
        )
        for layer_id, counts in layer_counts.items()
    }


def policy_outcomes(
    events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    *,
    policy: PolicyName,
    capacity_per_layer: int,
) -> tuple[PolicyEventOutcome, ...]:
    """Return causal outcomes used by both simulation and replay reports."""

    trace = _validated_trace(events, capacity_per_layer)
    if policy == "reactive":
        return tuple(
            PolicyEventOutcome(event, (), event.selected_expert_ids)
            for event in trace
        )

    if policy == "static_hotset":
        residents = _static_residents(trace, capacity_per_layer)
        return tuple(
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
            for event in trace
        )

    if policy != "lru":
        raise ValueError(f"unsupported policy: {policy}")
    caches: defaultdict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    outcomes: list[PolicyEventOutcome] = []
    for event in trace:
        cache = caches[event.layer_id]
        required = frozenset(event.selected_expert_ids)
        ready: list[int] = []
        blocking: list[int] = []
        for expert_id in event.selected_expert_ids:
            if expert_id in cache:
                ready.append(expert_id)
                cache.move_to_end(expert_id)
            else:
                blocking.append(expert_id)

        for expert_id in blocking:
            while len(cache) >= capacity_per_layer:
                victim = next(
                    cached for cached in cache if cached not in required
                )
                del cache[victim]
            cache[expert_id] = None
        outcomes.append(
            PolicyEventOutcome(event, tuple(ready), tuple(blocking))
        )
    return tuple(outcomes)


def simulate_policies(
    events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    *,
    capacity_per_layer: int,
) -> CacheSimulationReport:
    """Compare no-residency, trace hotset, and online LRU policies."""

    trace = _validated_trace(events, capacity_per_layer)

    demand_count = sum(len(event.selected_expert_ids) for event in trace)
    reactive = _result(
        "reactive",
        demand_count=demand_count,
        hit_count=0,
        load_count=demand_count,
    )

    static_residents = _static_residents(trace, capacity_per_layer)
    static_outcomes = policy_outcomes(
        trace, policy="static_hotset", capacity_per_layer=capacity_per_layer
    )
    static_hits = sum(len(outcome.ready_expert_ids) for outcome in static_outcomes)
    static_preloads = sum(
        len(residents) for residents in static_residents.values()
    )
    static_hotset = _result(
        "static_hotset",
        demand_count=demand_count,
        hit_count=static_hits,
        load_count=demand_count - static_hits,
        preload_count=static_preloads,
    )

    lru_outcomes = policy_outcomes(
        trace, policy="lru", capacity_per_layer=capacity_per_layer
    )
    lru_hits = sum(len(outcome.ready_expert_ids) for outcome in lru_outcomes)
    lru_loads = sum(
        len(outcome.blocking_expert_ids) for outcome in lru_outcomes
    )

    lru = _result(
        "lru",
        demand_count=demand_count,
        hit_count=lru_hits,
        load_count=lru_loads,
    )
    return CacheSimulationReport(
        measurement_kind="estimated",
        capacity_per_layer=capacity_per_layer,
        reactive=reactive,
        static_hotset=static_hotset,
        lru=lru,
    )
