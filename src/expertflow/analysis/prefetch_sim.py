"""Estimated concurrent-slot prefetch simulation over measured router traces."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Literal

from expertflow.trace.schema import RouterTraceEvent


PredictionMode = Literal["none", "oracle", "frequency"]

_PREDICTION_MODES = frozenset({"none", "oracle", "frequency"})


@dataclass(frozen=True, slots=True)
class PrefetchStepOutcome:
    """One interleaved decode/prefill step across all simulated slots."""

    step_index: int
    demand_count: int
    hit_count: int
    miss_count: int
    transfers_issued: int
    predicted_candidates: int
    ready_useful: int
    wasted_transfers: int


@dataclass(frozen=True, slots=True)
class PrefetchSimulationReport:
    """Estimated outcome of bounded next-step prefetch under slot capacity."""

    measurement_kind: Literal["estimated"]
    prediction: str
    capacity_per_layer: int
    max_transfers_per_step: int
    expert_transfer_ms: float
    slot_bytes: int
    conversation_count: int
    step_count: int
    demand_count: int
    hit_count: int
    miss_count: int
    hit_rate: float
    transfers_issued: int
    ready_useful: int
    wasted_transfers: int
    wasted_transfer_mib: float
    estimated_saved_blocking_ms: float
    steps: tuple[PrefetchStepOutcome, ...]


def _validated_inputs(
    traces: tuple[tuple[RouterTraceEvent, ...], ...],
    *,
    prediction: str,
    capacity_per_layer: int,
    max_transfers_per_step: int,
    expert_transfer_ms: float,
    slot_bytes: int,
) -> None:
    if prediction not in _PREDICTION_MODES:
        raise ValueError(
            "prediction must be 'none', 'oracle', or 'frequency'"
        )
    if not traces or any(not trace for trace in traces):
        raise ValueError("traces must be a non-empty sequence of non-empty traces")
    if any(
        len(event.selected_expert_ids) > capacity_per_layer
        for trace in traces
        for event in trace
    ):
        raise ValueError("capacity_per_layer cannot be below router top-k")
    if max_transfers_per_step < 0:
        raise ValueError("max_transfers_per_step must be non-negative")
    if expert_transfer_ms < 0:
        raise ValueError("expert_transfer_ms must be non-negative")
    if slot_bytes <= 0:
        raise ValueError("slot_bytes must be positive")


def _forward_groups(
    trace: tuple[RouterTraceEvent, ...],
) -> list[tuple[RouterTraceEvent, ...]]:
    groups: OrderedDict[int, list[RouterTraceEvent]] = OrderedDict()
    for event in trace:
        groups.setdefault(event.forward_id, []).append(event)
    return [
        tuple(
            sorted(
                events,
                key=lambda item: (
                    item.layer_id,
                    item.token_index,
                    item.hook_order,
                ),
            )
        )
        for events in groups.values()
    ]


def _consume_demand(
    cache: OrderedDict[int, None],
    event: RouterTraceEvent,
    *,
    capacity_per_layer: int,
    issued: set[int],
    counted_ready: set[tuple[int, int]],
) -> tuple[int, int]:
    required = set(event.selected_expert_ids)
    hits = 0
    misses = 0
    for expert_id in event.selected_expert_ids:
        if expert_id in cache:
            hits += 1
            cache.move_to_end(expert_id)
            if expert_id in issued and (
                event.layer_id,
                expert_id,
            ) not in counted_ready:
                counted_ready.add((event.layer_id, expert_id))
        else:
            misses += 1
            while len(cache) >= capacity_per_layer:
                victim = next(
                    cached for cached in cache if cached not in required
                )
                del cache[victim]
            cache[expert_id] = None
    return hits, misses


def simulate_concurrent_prefetch(
    traces: list[list[RouterTraceEvent]] | tuple[list[RouterTraceEvent], ...],
    *,
    prediction: PredictionMode,
    capacity_per_layer: int,
    max_transfers_per_step: int,
    expert_transfer_ms: float,
    slot_bytes: int,
) -> PrefetchSimulationReport:
    """Interleave conversations into shared-cache steps and estimate prefetch."""

    normalized = tuple(tuple(trace) for trace in traces)
    _validated_inputs(
        normalized,
        prediction=prediction,
        capacity_per_layer=capacity_per_layer,
        max_transfers_per_step=max_transfers_per_step,
        expert_transfer_ms=expert_transfer_ms,
        slot_bytes=slot_bytes,
    )

    caches: dict[int, OrderedDict[int, None]] = {}
    frequencies: dict[int, Counter[int]] = {}

    def cache_for(layer_id: int) -> OrderedDict[int, None]:
        return caches.setdefault(layer_id, OrderedDict())

    def frequencies_for(layer_id: int) -> Counter[int]:
        return frequencies.setdefault(layer_id, Counter())

    grouped = [_forward_groups(tuple(trace)) for trace in normalized]
    step_count = max(len(groups) for groups in grouped)

    total_demands = 0
    total_hits = 0
    total_misses = 0
    total_issued = 0
    total_ready_useful = 0
    total_wasted = 0
    total_predicted_candidates = 0
    step_outcomes: list[PrefetchStepOutcome] = []

    for step_index in range(step_count):
        step_events = [
            event
            for groups in grouped
            if step_index < len(groups)
            for event in groups[step_index]
        ]

        issued_layers: dict[int, set[int]] = {}
        candidates_total = 0
        if prediction != "none":
            if prediction == "frequency":
                scored = [
                    (-frequencies_for(layer_id)[expert_id], layer_id, expert_id)
                    for layer_id in caches
                    for expert_id in frequencies_for(layer_id)
                    if frequencies_for(layer_id)[expert_id] > 0
                    and expert_id not in cache_for(layer_id)
                ]
                scored.sort()
            else:
                step_demand: Counter[tuple[int, int]] = Counter()
                for event in step_events:
                    for expert_id in event.selected_expert_ids:
                        step_demand[(event.layer_id, expert_id)] += 1
                scored = []
                for (layer_id, expert_id), count in step_demand.items():
                    if expert_id in cache_for(layer_id):
                        continue
                    scored.append((-count, layer_id, expert_id))
                scored.sort()
            candidates_total = len(scored)
            budget = max_transfers_per_step
            for _, layer_id, expert_id in scored:
                if budget <= 0:
                    break
                cache = cache_for(layer_id)
                if expert_id in cache:
                    continue
                protected = set(issued_layers.get(layer_id, set()))
                protected.update(
                    expert
                    for event in step_events
                    if event.layer_id == layer_id
                    for expert in event.selected_expert_ids
                )
                if len(cache) < capacity_per_layer:
                    cache[expert_id] = None
                    issued_layers.setdefault(layer_id, set()).add(expert_id)
                else:
                    victim = next(
                        (
                            cached
                            for cached in cache
                            if cached not in protected
                        ),
                        None,
                    )
                    if victim is None:
                        continue
                    del cache[victim]
                    cache[expert_id] = None
                    issued_layers.setdefault(layer_id, set()).add(expert_id)
                budget -= 1

        issued_ids = {
            (layer_id, expert_id)
            for layer_id, experts in issued_layers.items()
            for expert_id in experts
        }
        step_issued = len(issued_ids)

        step_hits = 0
        step_misses = 0
        counted_ready: set[int] = set()
        for event in step_events:
            cache = cache_for(event.layer_id)
            event_issued = issued_layers.get(event.layer_id, set())
            hits, misses = _consume_demand(
                cache,
                event,
                capacity_per_layer=capacity_per_layer,
                issued=event_issued,
                counted_ready=counted_ready,
            )
            step_hits += hits
            step_misses += misses
            if prediction == "frequency":
                frequencies_for(event.layer_id).update(
                    event.selected_expert_ids
                )

        step_demands = sum(len(event.selected_expert_ids) for event in step_events)
        ready_useful = len(counted_ready)
        wasted = step_issued - ready_useful
        total_demands += step_demands
        total_hits += step_hits
        total_misses += step_misses
        total_issued += step_issued
        total_ready_useful += ready_useful
        total_wasted += wasted
        total_predicted_candidates += candidates_total
        step_outcomes.append(
            PrefetchStepOutcome(
                step_index=step_index,
                demand_count=step_demands,
                hit_count=step_hits,
                miss_count=step_misses,
                transfers_issued=step_issued,
                predicted_candidates=candidates_total,
                ready_useful=ready_useful,
                wasted_transfers=wasted,
            )
        )

    return PrefetchSimulationReport(
        measurement_kind="estimated",
        prediction=prediction,
        capacity_per_layer=capacity_per_layer,
        max_transfers_per_step=max_transfers_per_step,
        expert_transfer_ms=expert_transfer_ms,
        slot_bytes=slot_bytes,
        conversation_count=len(normalized),
        step_count=step_count,
        demand_count=total_demands,
        hit_count=total_hits,
        miss_count=total_misses,
        hit_rate=total_hits / total_demands if total_demands else 0.0,
        transfers_issued=total_issued,
        ready_useful=total_ready_useful,
        wasted_transfers=total_wasted,
        wasted_transfer_mib=round(total_wasted * slot_bytes / (1024**2), 6),
        estimated_saved_blocking_ms=round(
            total_ready_useful * expert_transfer_ms, 6
        ),
        steps=tuple(step_outcomes),
    )
