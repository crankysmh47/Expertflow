"""VRAM-capacity and transfer-cost curves over measured routing events."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict

from expertflow.analysis.cache_sim import PolicyResult, simulate_policies
from expertflow.trace.schema import RouterTraceEvent


def _policy_point(
    result: PolicyResult,
    *,
    layer_count: int,
    router_top_k: int,
    expert_transfer_ms: float,
) -> dict[str, object]:
    body: dict[str, object] = asdict(result)
    body["estimated_serial_h2d_ms_per_layer_sweep"] = (
        layer_count
        * router_top_k
        * (1.0 - result.hit_rate)
        * expert_transfer_ms
    )
    return body


def build_capacity_curve(
    events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    *,
    capacities: tuple[int, ...],
    slot_bytes: int,
    expert_transfer_ms: float,
) -> dict[str, object]:
    """Fit policy curves on one ordered workload and label them in-sample."""

    trace = tuple(events)
    if not trace:
        raise ValueError("events must not be empty")
    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise ValueError("capacities must contain positive values")
    if slot_bytes <= 0:
        raise ValueError("slot_bytes must be positive")
    if expert_transfer_ms <= 0:
        raise ValueError("expert_transfer_ms must be positive")

    top_k_values = {len(event.selected_expert_ids) for event in trace}
    if len(top_k_values) != 1:
        raise ValueError("router top-k must be constant across the workload")
    router_top_k = next(iter(top_k_values))
    layer_ids = sorted({event.layer_id for event in trace})
    points: list[dict[str, object]] = []
    for capacity in dict.fromkeys(capacities):
        simulation = simulate_policies(
            trace, capacity_per_layer=capacity
        )
        points.append(
            {
                "capacity_per_layer": capacity,
                "projected_cache_bytes": (
                    capacity * len(layer_ids) * slot_bytes
                ),
                "reactive": _policy_point(
                    simulation.reactive,
                    layer_count=len(layer_ids),
                    router_top_k=router_top_k,
                    expert_transfer_ms=expert_transfer_ms,
                ),
                "static_hotset": _policy_point(
                    simulation.static_hotset,
                    layer_count=len(layer_ids),
                    router_top_k=router_top_k,
                    expert_transfer_ms=expert_transfer_ms,
                ),
                "lru": _policy_point(
                    simulation.lru,
                    layer_count=len(layer_ids),
                    router_top_k=router_top_k,
                    expert_transfer_ms=expert_transfer_ms,
                ),
            }
        )

    return {
        "schema_version": "1.0.0",
        "measurement_kind": "estimated",
        "fit_scope": "in_sample",
        "event_count": len(trace),
        "expert_demand_count": sum(
            len(event.selected_expert_ids) for event in trace
        ),
        "layer_ids": layer_ids,
        "router_top_k": router_top_k,
        "slot_bytes": slot_bytes,
        "expert_transfer_ms": expert_transfer_ms,
        "points": points,
    }


def build_held_out_capacity_curve(
    training_events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    evaluation_events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    *,
    capacities: tuple[int, ...],
    slot_bytes: int,
    expert_transfer_ms: float,
) -> dict[str, object]:
    """Fit static residents on training events and score untouched events."""

    training = tuple(training_events)
    evaluation = tuple(evaluation_events)
    if not training or not evaluation:
        raise ValueError("training and evaluation events must not be empty")
    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise ValueError("capacities must contain positive values")
    if slot_bytes <= 0 or expert_transfer_ms <= 0:
        raise ValueError("slot bytes and transfer time must be positive")

    top_k_values = {
        len(event.selected_expert_ids)
        for event in (*training, *evaluation)
    }
    if len(top_k_values) != 1:
        raise ValueError("router top-k must be constant across both workloads")
    router_top_k = next(iter(top_k_values))
    training_layers = {event.layer_id for event in training}
    evaluation_layers = {event.layer_id for event in evaluation}
    if training_layers != evaluation_layers:
        raise ValueError("training and evaluation layer sets must match")
    layer_ids = sorted(evaluation_layers)
    demand_count = sum(
        len(event.selected_expert_ids) for event in evaluation
    )

    counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for event in training:
        counts[event.layer_id].update(event.selected_expert_ids)

    points: list[dict[str, object]] = []
    for capacity in dict.fromkeys(capacities):
        if capacity < router_top_k:
            raise ValueError("capacity cannot be below router top-k")
        residents = {
            layer_id: frozenset(
                expert_id
                for expert_id, _ in sorted(
                    layer_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:capacity]
            )
            for layer_id, layer_counts in counts.items()
        }
        static_hits = sum(
            expert_id in residents[event.layer_id]
            for event in evaluation
            for expert_id in event.selected_expert_ids
        )
        static = PolicyResult(
            policy="static_hotset",
            demand_count=demand_count,
            hit_count=static_hits,
            miss_count=demand_count - static_hits,
            load_count=demand_count - static_hits,
            preload_count=sum(len(values) for values in residents.values()),
            hit_rate=static_hits / demand_count,
        )
        evaluation_simulation = simulate_policies(
            evaluation, capacity_per_layer=capacity
        )
        points.append(
            {
                "capacity_per_layer": capacity,
                "projected_cache_bytes": (
                    capacity * len(layer_ids) * slot_bytes
                ),
                "reactive": _policy_point(
                    evaluation_simulation.reactive,
                    layer_count=len(layer_ids),
                    router_top_k=router_top_k,
                    expert_transfer_ms=expert_transfer_ms,
                ),
                "static_hotset": _policy_point(
                    static,
                    layer_count=len(layer_ids),
                    router_top_k=router_top_k,
                    expert_transfer_ms=expert_transfer_ms,
                ),
                "lru": _policy_point(
                    evaluation_simulation.lru,
                    layer_count=len(layer_ids),
                    router_top_k=router_top_k,
                    expert_transfer_ms=expert_transfer_ms,
                ),
            }
        )

    return {
        "schema_version": "1.0.0",
        "measurement_kind": "estimated",
        "fit_scope": "held_out",
        "training_event_count": len(training),
        "evaluation_event_count": len(evaluation),
        "event_count": len(evaluation),
        "expert_demand_count": demand_count,
        "layer_ids": layer_ids,
        "router_top_k": router_top_k,
        "slot_bytes": slot_bytes,
        "expert_transfer_ms": expert_transfer_ms,
        "points": points,
    }
