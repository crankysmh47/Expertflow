"""Frozen workload construction and per-layer Q6 routing-locality metrics."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from expertflow.quality.mmlu import build_zero_shot_prompt
from expertflow.trace.schema import RouterTraceEvent


@dataclass(frozen=True, slots=True)
class RoutingWorkload:
    workload_id: str
    workload_family: str
    prompt: str
    n_predict: int
    split: str | None = None
    domain: str | None = None
    source_offset: int | None = None


def merged_workload_ids(
    saved: Sequence[Mapping[str, Any]], workloads: Sequence[RoutingWorkload]
) -> tuple[str, ...]:
    """Keep resume metadata complete while collecting families separately."""

    return tuple(
        dict.fromkeys(
            [str(row["workload_id"]) for row in saved]
            + [row.workload_id for row in workloads]
        )
    )


def probe_command(
    workload: RoutingWorkload,
    *,
    probe: str,
    model: str,
    tokens: str,
    trace: str,
) -> list[str]:
    """Render the frozen cache-disabled canonical-observer command."""

    prompt = (
        "<start_of_turn>user\n"
        f"{workload.prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    return [
        probe,
        "-m",
        model,
        "--tokens",
        tokens,
        "--trace",
        trace,
        "--trace-mode",
        "full",
        "-n",
        str(workload.n_predict),
        "-ngl",
        "10",
        "--threads",
        "12",
        prompt,
    ]


def simulate_hybrid_candidate(
    *,
    static_tps: float,
    cached_layers: Sequence[int],
    capacity: int,
    full_slots: int,
    shadow_bytes_per_layer: int,
    misses_per_token: Mapping[int, float],
    transfer_ms_per_miss: float,
    intrinsic_overhead_fraction_per_layer: float,
) -> dict[str, Any]:
    """Project one hybrid using measured misses and prior live-cache overhead."""

    if not 0 < capacity <= full_slots:
        raise ValueError("capacity must be within the full slot count")
    static_ms = 1000.0 / static_tps
    blocking_ms = sum(misses_per_token[layer] for layer in cached_layers) * transfer_ms_per_miss
    intrinsic_ms = static_ms * intrinsic_overhead_fraction_per_layer * len(cached_layers)
    projected_tps = 1000.0 / (static_ms + blocking_ms + intrinsic_ms)
    freed_bytes = round(
        shadow_bytes_per_layer * (full_slots - capacity) / full_slots
    ) * len(cached_layers)
    return {
        "static_layers": [],
        "cached_layers": list(cached_layers),
        "capacity": capacity,
        "freed_bytes": freed_bytes,
        "freed_mib": freed_bytes / (1024 * 1024),
        "misses_per_token": sum(misses_per_token[layer] for layer in cached_layers),
        "blocking_ms_per_token": blocking_ms,
        "intrinsic_overhead_ms_per_token": intrinsic_ms,
        "projected_tps": projected_tps,
        "retained_tps_fraction": projected_tps / static_tps,
    }


def build_workloads(
    *,
    ppl_text: str,
    mmlu_items: Sequence[Mapping[str, Any]],
    conversations: Sequence[Mapping[str, Any]],
    ppl_window_count: int = 8,
    ppl_window_chars: int = 2048,
) -> tuple[RoutingWorkload, ...]:
    """Build the four frozen workload families without random selection."""

    if ppl_window_count <= 0 or ppl_window_chars <= 0:
        raise ValueError("PPL window count and size must be positive")
    if len(ppl_text) < ppl_window_chars:
        raise ValueError("PPL corpus is shorter than one requested window")

    workloads = [
        RoutingWorkload("performance-512", "performance_512", "Caching.", 512)
    ]
    span = len(ppl_text) - ppl_window_chars
    offsets = (
        [0]
        if ppl_window_count == 1
        else [round(index * span / (ppl_window_count - 1)) for index in range(ppl_window_count)]
    )
    for index, offset in enumerate(offsets):
        workloads.append(
            RoutingWorkload(
                f"ppl-window-{index + 1:02d}",
                "heldout_ppl",
                ppl_text[offset : offset + ppl_window_chars],
                1,
                source_offset=offset,
            )
        )
    for index, item in enumerate(mmlu_items):
        workloads.append(
            RoutingWorkload(
                f"mmlu-{index:03d}",
                "fixed_mmlu",
                build_zero_shot_prompt(item),
                1,
                domain=str(item["subject"]),
            )
        )

    seen: set[tuple[str, str]] = set()
    for row in conversations:
        key = str(row["split"]), str(row["domain"])
        if key in seen:
            continue
        seen.add(key)
        workloads.append(
            RoutingWorkload(
                str(row["conversation_id"]),
                "representative_conversation",
                str(row["prompt"]),
                32,
                split=key[0],
                domain=key[1],
            )
        )
    return tuple(workloads)


def _reuse_distance(events: Iterable[RouterTraceEvent]) -> dict[str, Any]:
    stack: list[int] = []
    cold = 0
    distances: Counter[int] = Counter()
    for event in events:
        for expert in event.selected_expert_ids:
            if expert not in stack:
                cold += 1
            else:
                index = stack.index(expert)
                distances[index] += 1
                stack.pop(index)
            stack.insert(0, expert)
    expanded = sorted(distance for distance, count in distances.items() for _ in range(count))
    return {
        "cold": cold,
        "reuse_count": len(expanded),
        "histogram": {str(distance): count for distance, count in sorted(distances.items())},
        "p50": expanded[len(expanded) // 2] if expanded else None,
        "p95": expanded[min(len(expanded) - 1, int(len(expanded) * 0.95))] if expanded else None,
    }


def _lru(events_by_sequence: Mapping[str, Sequence[RouterTraceEvent]], capacity: int) -> tuple[int, int]:
    hits = 0
    misses = 0
    for events in events_by_sequence.values():
        cache: OrderedDict[int, None] = OrderedDict()
        for event in events:
            required = frozenset(event.selected_expert_ids)
            missing: list[int] = []
            for expert in event.selected_expert_ids:
                if expert in cache:
                    hits += 1
                    cache.move_to_end(expert)
                else:
                    misses += 1
                    missing.append(expert)
            for expert in missing:
                while len(cache) >= capacity:
                    victim = next(candidate for candidate in cache if candidate not in required)
                    del cache[victim]
                cache[expert] = None
    return hits, misses


def summarize_routing_locality(
    traces: Mapping[str, Sequence[RouterTraceEvent]],
    *,
    capacities: Sequence[int] = (112, 96, 80, 64),
    expert_bundle_bytes: int = 5_358_852,
) -> dict[int, dict[str, Any]]:
    """Summarize demand locality per layer, resetting online state per sequence."""

    by_layer: defaultdict[int, dict[str, list[RouterTraceEvent]]] = defaultdict(dict)
    for sequence, events in traces.items():
        for event in events:
            by_layer[event.layer_id].setdefault(sequence, []).append(event)

    report: dict[int, dict[str, Any]] = {}
    for layer, sequences in sorted(by_layer.items()):
        ordered_events = [event for events in sequences.values() for event in events]
        frequency = Counter(expert for event in ordered_events for expert in event.selected_expert_ids)
        unique_counts = [len({expert for event in events for expert in event.selected_expert_ids}) for events in sequences.values()]
        adjacent_reused = 0
        adjacent_demands = 0
        for events in sequences.values():
            for previous, current in zip(events, events[1:]):
                adjacent_reused += len(set(previous.selected_expert_ids) & set(current.selected_expert_ids))
                adjacent_demands += len(current.selected_expert_ids)
        lru: dict[str, Any] = {}
        for capacity in capacities:
            hits, misses = _lru(sequences, capacity)
            event_count = len(ordered_events)
            lru[str(capacity)] = {
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
                "misses_per_token": misses / event_count if event_count else 0.0,
                "h2d_bytes_per_token": misses * expert_bundle_bytes / event_count if event_count else 0.0,
            }
        reuse_parts = [_reuse_distance(events) for events in sequences.values()]
        histogram: Counter[int] = Counter()
        for part in reuse_parts:
            histogram.update({int(distance): count for distance, count in part["histogram"].items()})
        expanded = sorted(distance for distance, count in histogram.items() for _ in range(count))
        cold_count = sum(part["cold"] for part in reuse_parts)
        demand_count = sum(frequency.values())
        report[layer] = {
            "event_count": len(ordered_events),
            "expert_demand_count": demand_count,
            "frequency": {str(expert): count for expert, count in sorted(frequency.items())},
            "working_set_size": len(frequency),
            "unique_experts_per_sequence": {
                "min": min(unique_counts),
                "max": max(unique_counts),
                "mean": mean(unique_counts),
            },
            "adjacent_token_reuse_rate": adjacent_reused / adjacent_demands if adjacent_demands else 0.0,
            "temporal_reuse_rate": (demand_count - cold_count) / demand_count if demand_count else 0.0,
            "reuse_distance": {
                "cold": cold_count,
                "reuse_count": len(expanded),
                "histogram": {str(distance): count for distance, count in sorted(histogram.items())},
                "p50": expanded[len(expanded) // 2] if expanded else None,
                "p95": expanded[min(len(expanded) - 1, int(len(expanded) * 0.95))] if expanded else None,
            },
            "lru": lru,
        }
    return report
