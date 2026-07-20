from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from expertflow.benchmark.performance import parse_probe_result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_t2_run(
    performance_path: Path,
    cache_path: Path,
    sidecar_path: Path | None,
) -> dict[str, Any]:
    result = parse_probe_result(performance_path)
    cache_events = _jsonl(cache_path)
    demands = 0
    hits = 0
    misses = 0
    sidecar_demands = 0
    reactive_bytes = 0
    reactive_blocking_us = 0
    for event in cache_events:
        selected = event.get("selected", event.get("selected_expert_ids", []))
        physical = event.get("physical_slots", [])
        demands += len(selected)
        hits += int(event["hits"])
        misses += int(event.get("blocking_misses", event.get("misses", 0)))
        sidecar_demands += sum(int(slot) >= 32 for slot in physical)
        reactive_bytes += int(event["bytes_transferred"])
        reactive_blocking_us += int(event["blocking_duration_us"])
    if hits + misses + sidecar_demands != demands:
        raise ValueError(
            "cache demands do not reconcile: "
            f"{hits}+{misses}+{sidecar_demands}!={demands}"
        )

    records: list[dict[str, Any]] = []
    runtime_summary: dict[str, Any] = {}
    if sidecar_path is not None:
        for value in _jsonl(sidecar_path):
            if value.get("record_kind") == "summary":
                runtime_summary = value
            elif value.get("record_kind") == "prefetch":
                records.append(value)
            else:
                raise ValueError("unknown T2 record kind")
        if int(runtime_summary.get("records", -1)) != len(records):
            raise ValueError("T2 runtime summary record count does not match")

    enqueued = [record for record in records if record["transfer_enqueued"]]
    ready = [record for record in records if record["outcome"] == "ready_useful"]
    late = [record for record in records if record["outcome"] == "late_useful"]
    wasted = [record for record in records if record["outcome"] == "wasted"]
    if runtime_summary:
        expected = {
            "enqueued": len(enqueued),
            "ready_useful": len(ready),
            "late_useful": len(late),
            "wasted": len(wasted),
        }
        for key, value in expected.items():
            if int(runtime_summary.get(key, -1)) != value:
                raise ValueError(f"T2 runtime summary {key} does not match")

    sidecar_blocking_us = sum(int(record["blocking_wait_us"]) for record in records)
    result.update(
        {
            "cache_events": len(cache_events),
            "expert_demands": demands,
            "reactive_hits": hits,
            "blocking_misses": misses,
            "sidecar_demands": sidecar_demands,
            "reactive_hit_rate": hits / demands if demands else None,
            "reactive_bytes_transferred": reactive_bytes,
            "reactive_blocking_ms": reactive_blocking_us / 1000.0,
            "sidecar_blocking_ms": sidecar_blocking_us / 1000.0,
            "total_blocking_ms": (
                reactive_blocking_us + sidecar_blocking_us
            )
            / 1000.0,
            "prefetch_records": len(records),
            "prefetches_enqueued": len(enqueued),
            "ready_useful_prefetches": len(ready),
            "late_useful_prefetches": len(late),
            "wasted_prefetches": len(wasted),
            "prefetch_bytes": sum(int(record["bytes"]) for record in enqueued),
            "wasted_bytes": sum(int(record["bytes"]) for record in wasted),
            "staging_ms": sum(int(record["staging_ns"]) for record in enqueued)
            / 1_000_000.0,
            "enqueue_ms": sum(int(record["enqueue_ns"]) for record in enqueued)
            / 1_000_000.0,
            "queue_to_ready_ms": sum(
                int(record["queue_to_ready_ns"]) for record in enqueued
            )
            / 1_000_000.0,
            "h2d_cuda_event_ms": sum(
                float(record["h2d_cuda_event_ms"]) for record in enqueued
            ),
            "arena_bytes": (
                int(runtime_summary["arena_bytes"])
                if "arena_bytes" in runtime_summary
                else None
            ),
        }
    )
    return result


def analyze_prevented_misses(
    reactive_cache_path: Path,
    predictive_cache_path: Path,
) -> dict[str, int]:
    reactive_events = _jsonl(reactive_cache_path)
    predictive_events = _jsonl(predictive_cache_path)
    if len(reactive_events) != len(predictive_events):
        raise ValueError("cache event counts differ")

    paired_sidecar_demands = 0
    prevented = 0
    redundant = 0
    for reactive, predictive in zip(
        reactive_events, predictive_events, strict=True
    ):
        reactive_identity = (
            reactive.get("token_index"),
            reactive.get("layer_id"),
        )
        predictive_identity = (
            predictive.get("token_index"),
            predictive.get("layer_id"),
        )
        if reactive_identity != predictive_identity:
            raise ValueError("cache event identities differ")
        reactive_selected = reactive.get(
            "selected", reactive.get("selected_expert_ids", [])
        )
        predictive_selected = predictive.get(
            "selected", predictive.get("selected_expert_ids", [])
        )
        if reactive_selected != predictive_selected:
            raise ValueError("router selections differ")
        predictive_physical = predictive.get("physical_slots", [])
        if len(predictive_physical) != len(predictive_selected):
            raise ValueError("predictive physical-slot mapping is incomplete")
        reactive_loads = {
            int(load["expert"]) for load in reactive.get("loads", [])
        }
        for expert_id, slot_id in zip(
            predictive_selected, predictive_physical, strict=True
        ):
            if int(slot_id) < 32:
                continue
            paired_sidecar_demands += 1
            if int(expert_id) in reactive_loads:
                prevented += 1
            else:
                redundant += 1

    return {
        "paired_sidecar_demands": paired_sidecar_demands,
        "actual_blocking_misses_prevented": prevented,
        "sidecar_demands_that_were_reactive_hits": redundant,
    }


def compare_t2_pair(
    reactive: dict[str, Any],
    predictive: dict[str, Any],
) -> dict[str, float | None]:
    def change(key: str, higher_is_better: bool) -> float | None:
        before = reactive.get(key)
        after = predictive.get(key)
        if before in (None, 0) or after is None:
            return None
        raw = (float(after) - float(before)) / float(before) * 100.0
        return raw if higher_is_better else -raw

    return {
        "decode_tps_percent": change("decode_tps", True),
        "prompt_eval_tps_percent": change("prompt_eval_tps", True),
        "end_to_end_percent": change("end_to_end_ms", False),
        "blocking_miss_reduction_percent": change(
            "blocking_misses", False
        ),
        "blocking_time_reduction_percent": change(
            "total_blocking_ms", False
        ),
    }
