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


def simulate_policies(
    events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    *,
    capacity_per_layer: int,
) -> CacheSimulationReport:
    """Compare no-residency, trace hotset, and online LRU policies."""

    if capacity_per_layer <= 0:
        raise ValueError("capacity_per_layer must be positive")
    trace = tuple(events)
    if any(
        len(event.selected_expert_ids) > capacity_per_layer for event in trace
    ):
        raise ValueError("capacity_per_layer cannot be below router top-k")

    demand_count = sum(len(event.selected_expert_ids) for event in trace)
    reactive = _result(
        "reactive",
        demand_count=demand_count,
        hit_count=0,
        load_count=demand_count,
    )

    layer_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for event in trace:
        layer_counts[event.layer_id].update(event.selected_expert_ids)
    static_residents = {
        layer_id: frozenset(
            expert_id
            for expert_id, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:capacity_per_layer]
        )
        for layer_id, counts in layer_counts.items()
    }
    static_hits = sum(
        expert_id in static_residents[event.layer_id]
        for event in trace
        for expert_id in event.selected_expert_ids
    )
    static_preloads = sum(len(residents) for residents in static_residents.values())
    static_hotset = _result(
        "static_hotset",
        demand_count=demand_count,
        hit_count=static_hits,
        load_count=demand_count - static_hits,
        preload_count=static_preloads,
    )

    lru_caches: defaultdict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    lru_hits = 0
    lru_loads = 0
    for event in trace:
        cache = lru_caches[event.layer_id]
        required = frozenset(event.selected_expert_ids)
        misses: list[int] = []
        for expert_id in event.selected_expert_ids:
            if expert_id in cache:
                lru_hits += 1
                cache.move_to_end(expert_id)
            else:
                misses.append(expert_id)

        for expert_id in misses:
            while len(cache) >= capacity_per_layer:
                victim = next(
                    cached for cached in cache if cached not in required
                )
                del cache[victim]
            cache[expert_id] = None
            lru_loads += 1

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
