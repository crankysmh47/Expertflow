from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PACKED_EXPERT_BYTES = 3_345_412


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reconcile_multilayer_events(
    cache_path: Path,
    trace_path: Path,
    *,
    enabled_layers: Iterable[int],
    arena_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layers = tuple(enabled_layers)
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("enabled layers must be unique and ascending")
    events = _read_jsonl(cache_path)
    if len(events) % len(layers):
        raise ValueError("cache event count is not a complete layer cycle")

    per_layer = {
        str(layer): {
            "events": 0,
            "expert_demands": 0,
            "hits": 0,
            "misses": 0,
            "bytes_transferred": 0,
            "blocking_wall_ms": 0.0,
        }
        for layer in layers
    }
    cache_selected: list[tuple[int, list[int]]] = []
    previous_residency: dict[int, dict[int, dict[str, int]]] = {
        layer: {} for layer in layers
    }
    for index, event in enumerate(events):
        expected_layer = layers[index % len(layers)]
        layer = int(event["layer_id"])
        if layer != expected_layer:
            raise ValueError(
                f"cache layer order mismatch at {index}: {layer} != {expected_layer}"
            )
        selected = [int(value) for value in event.get("selected", [])]
        if len(selected) != 8 or len(set(selected)) != 8:
            raise ValueError("each cache event must contain eight unique demands")
        hits = int(event["hits"])
        misses = int(event.get("blocking_misses", event.get("misses", 0)))
        if hits + misses != len(selected):
            raise ValueError("cache hits and misses do not reconcile")
        transferred = int(event["bytes_transferred"])
        if transferred != misses * PACKED_EXPERT_BYTES:
            raise ValueError("cache bytes do not reconcile with packed misses")
        mapping_rows = event.get("final_resident_mapping", [])
        if mapping_rows:
            if len(mapping_rows) != 32:
                raise ValueError("resident mapping must contain exactly 32 slots")
            mapping = {int(row["slot"]): row for row in mapping_rows}
            if set(mapping) != set(range(32)):
                raise ValueError("resident mapping slots are incomplete or duplicated")
            physical_slots = [int(value) for value in event["physical_slots"]]
            if len(physical_slots) != len(selected):
                raise ValueError("physical slot count does not match selected experts")
            for expert, slot in zip(selected, physical_slots):
                row = mapping.get(slot)
                if row is None or int(row["expert"]) != expert:
                    raise ValueError("resident mapping does not match selected expert")
            prior = previous_residency[layer]
            for load in event.get("loads", []):
                slot = int(load["slot"])
                before = prior.get(
                    slot,
                    {"expert": -1, "generation": 0, "last_use_sequence": 0},
                )
                after = mapping[slot]
                if (
                    int(load["replaced"]) != int(before["expert"])
                    or int(load["generation_before"]) != int(before["generation"])
                    or int(load["generation_after"]) != int(after["generation"])
                    or int(load["expert"]) != int(after["expert"])
                ):
                    raise ValueError("resident mapping load generation is stale")
            previous_residency[layer] = mapping
        stats = per_layer[str(layer)]
        stats["events"] += 1
        stats["expert_demands"] += len(selected)
        stats["hits"] += hits
        stats["misses"] += misses
        stats["bytes_transferred"] += transferred
        stats["blocking_wall_ms"] += float(event["blocking_duration_us"]) / 1000.0
        cache_selected.append((layer, selected))

    trace_events = [
        event for event in _read_jsonl(trace_path)
        if int(event.get("layer_id", -1)) in layers
    ]
    if trace_events:
        trace_selected = [
            (
                int(event["layer_id"]),
                [int(value) for value in event["selected_expert_ids"]],
            )
            for event in trace_events
        ]
        if trace_selected != cache_selected:
            raise ValueError("cache and router event ordering or selected IDs differ")

    if arena_layout is not None:
        total = int(arena_layout["total_bytes"])
        previous_end = 0
        for layer in layers:
            entry = arena_layout["layers"][str(layer)]
            start = int(entry["gate_up_offset"])
            end = int(entry["end_offset"])
            if start < previous_end or end <= start or end > total:
                raise ValueError("arena layer regions overlap or exceed allocation")
            previous_end = end

    return {
        "events": len(events),
        "expert_demands": sum(value["expert_demands"] for value in per_layer.values()),
        "hits": sum(value["hits"] for value in per_layer.values()),
        "misses": sum(value["misses"] for value in per_layer.values()),
        "bytes_transferred": sum(value["bytes_transferred"] for value in per_layer.values()),
        "blocking_wall_ms": sum(value["blocking_wall_ms"] for value in per_layer.values()),
        "layers": per_layer,
    }
