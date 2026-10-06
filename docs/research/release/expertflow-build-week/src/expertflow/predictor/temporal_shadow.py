"""Conversation-reset 32-slot all-ready temporal prefetch simulation."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Iterable, Sequence

from expertflow.predictor.temporal_dataset import TemporalSample


def _demand(cache: OrderedDict[int, None], expert: int, capacity: int) -> bool:
    hit = expert in cache
    if hit:
        cache.move_to_end(expert)
    else:
        if len(cache) >= capacity:
            cache.popitem(last=False)
        cache[expert] = None
    return hit


def simulate_temporal_shadow(
    samples: Iterable[TemporalSample],
    rankings: Sequence[tuple[int, ...]],
    *,
    width: int,
    capacity: int = 32,
    expert_bytes: int = 3_345_412,
) -> dict[str, int | str]:
    rows = tuple(samples)
    if len(rows) != len(rankings):
        raise ValueError("samples and rankings must have identical length")
    if not 1 <= width <= 128 or capacity < width:
        raise ValueError("temporal width and capacity are incompatible")

    totals = defaultdict(int)
    current_conversation: str | None = None
    reactive: OrderedDict[int, None] = OrderedDict()
    shadow: OrderedDict[int, None] = OrderedDict()
    previous: TemporalSample | None = None
    for sample, ranking in zip(rows, rankings, strict=True):
        if sample.conversation_id != current_conversation:
            current_conversation = sample.conversation_id
            reactive.clear()
            shadow.clear()
            previous = None
            for expert in sample.source_expert_ids:
                _demand(reactive, expert, capacity)
                _demand(shadow, expert, capacity)
        elif previous is not None and sample.source_expert_ids != previous.target_expert_ids:
            raise ValueError("temporal shadow input is not a continuous conversation sequence")

        target = set(sample.target_expert_ids)
        totals["demand_count"] += len(target)
        reactive_hits = sum(expert in reactive for expert in target)
        totals["reactive_demand_hits"] += reactive_hits

        inserted: set[int] = set()
        for expert in ranking[:width]:
            if expert in shadow:
                shadow.move_to_end(expert)
                continue
            if len(shadow) >= capacity:
                victim, _ = shadow.popitem(last=False)
                totals["additional_speculative_evictions"] += 1
                if victim in target:
                    totals["eviction_regret"] += 1
            shadow[expert] = None
            inserted.add(expert)
            if expert in target:
                totals["useful_speculative_insertions"] += 1
            else:
                totals["wasted_speculative_insertions"] += 1

        ready = sum(expert in shadow for expert in target)
        totals["ready_before_demand"] += ready
        totals["uncovered_blocking_misses"] += len(target) - ready
        for expert in sample.target_expert_ids:
            _demand(reactive, expert, capacity)
            _demand(shadow, expert, capacity)
        previous = sample

    totals["reactive_blocking_misses"] = totals["demand_count"] - totals["reactive_demand_hits"]
    totals["ready_improvement_over_reactive"] = (
        totals["ready_before_demand"] - totals["reactive_demand_hits"]
    )
    totals["useful_predicted_bytes"] = totals["useful_speculative_insertions"] * expert_bytes
    totals["wasted_predicted_bytes"] = totals["wasted_speculative_insertions"] * expert_bytes
    for metric in (
        "eviction_regret",
        "additional_speculative_evictions",
        "useful_speculative_insertions",
        "wasted_speculative_insertions",
        "uncovered_blocking_misses",
    ):
        totals[metric] += 0
    return {
        **dict(totals),
        "measurement_kind": "simulated_all_ready_temporal_shadow",
        "candidate_width": width,
        "capacity": capacity,
        "expert_bytes": expert_bytes,
        "timing_model": "predictions are assumed ready before next-token demand",
    }
