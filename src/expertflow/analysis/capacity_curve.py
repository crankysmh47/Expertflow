"""VRAM-capacity and transfer-cost curves over measured routing events."""

from __future__ import annotations

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
