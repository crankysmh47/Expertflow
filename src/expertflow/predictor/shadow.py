"""Conversation-reset 32-slot speculative shadow simulation."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Iterable, Sequence

from expertflow.predictor.dataset import PredictionSample


def _demand(cache: OrderedDict[int, None], expert: int, capacity: int) -> bool:
    hit = expert in cache
    if hit:
        cache.move_to_end(expert)
    else:
        if len(cache) >= capacity:
            cache.popitem(last=False)
        cache[expert] = None
    return hit


def simulate_shadow(
    samples: Iterable[PredictionSample],
    rankings: Sequence[tuple[int, ...]],
    *,
    width: int,
    capacity: int = 32,
    expert_bytes: int = 3_345_412,
) -> dict[str, int | float | str]:
    rows = tuple(samples)
    if len(rows) != len(rankings):
        raise ValueError("samples and rankings must have identical length")
    if width not in {8, 12, 16} and not (1 <= width <= 128):
        raise ValueError("candidate width is invalid")
    if capacity < width:
        raise ValueError("cache capacity must be at least the candidate width")

    reactive: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    shadow: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    current_conversation: str | None = None
    totals = defaultdict(int)

    for sample, ranking in zip(rows, rankings, strict=True):
        if sample.conversation_id != current_conversation:
            reactive.clear()
            shadow.clear()
            current_conversation = sample.conversation_id
        target = set(sample.target_expert_ids)
        layer_reactive = reactive[sample.target_layer]
        layer_shadow = shadow[sample.target_layer]
        pre_speculation = set(layer_shadow)
        totals["demand_count"] += len(target)
        totals["reactive_demand_hits"] += sum(expert in layer_reactive for expert in target)

        inserted: set[int] = set()
        for expert in ranking[:width]:
            if expert in layer_shadow:
                continue
            victim = None
            if len(layer_shadow) >= capacity:
                victim, _ = layer_shadow.popitem(last=False)
                totals["additional_speculative_evictions"] += 1
                if victim in target:
                    totals["eviction_regret"] += 1
            layer_shadow[expert] = None
            inserted.add(expert)
            if expert in target:
                totals["useful_speculative_insertions"] += 1
            else:
                totals["wasted_speculative_insertions"] += 1

        totals["ready_before_demand"] += sum(expert in layer_shadow for expert in target)
        totals["predicted_ready_demands"] += sum(
            expert in target and expert in ranking[:width] and expert not in pre_speculation
            for expert in target
        )
        totals["uncovered_blocking_misses"] += sum(expert not in layer_shadow for expert in target)
        for expert in sample.target_expert_ids:
            _demand(layer_reactive, expert, capacity)
            _demand(layer_shadow, expert, capacity)

    totals["late_predictions"] = 0
    totals["useful_predicted_bytes"] = totals["useful_speculative_insertions"] * expert_bytes
    totals["wasted_predicted_bytes"] = totals["wasted_speculative_insertions"] * expert_bytes
    totals["reactive_blocking_misses"] = totals["demand_count"] - totals["reactive_demand_hits"]
    totals["ready_improvement_over_reactive"] = totals["ready_before_demand"] - totals["reactive_demand_hits"]
    result: dict[str, int | float | str] = dict(totals)
    result.update({
        "measurement_kind": "simulated_shadow",
        "candidate_width": width,
        "capacity_per_layer": capacity,
        "expert_bytes": expert_bytes,
        "timing_model": "all predictions ready; no transfer/compute timing modeled",
    })
    return result
