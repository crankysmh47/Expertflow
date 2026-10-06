from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def parse_probe_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    prompt_ms = float(result["prompt_eval_ms"])
    decode_ms = float(result["decode_eval_ms"])
    prompt_tokens = int(result["prompt_tokens"])
    generated_tokens = int(result["generated_tokens"])
    latencies = [float(value) for value in result.get("decode_token_latencies_ms", [])]
    result["prompt_eval_tps"] = prompt_tokens * 1000.0 / prompt_ms if prompt_ms else None
    result["decode_tps"] = generated_tokens * 1000.0 / decode_ms if decode_ms else None
    result["token_latency_p50_ms"] = _percentile(latencies, 0.50)
    result["token_latency_p95_ms"] = _percentile(latencies, 0.95)
    return result


def summarize_repetitions(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        set.intersection(
            *(set(key for key, value in run.items() if isinstance(value, (int, float)))
              for run in repetitions)
        )
    ) if repetitions else []
    summary: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(run[key]) for run in repetitions]
        summary[key] = {
            "values": values,
            "mean": statistics.fmean(values),
            "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        }
    return summary


def parse_cache_events(path: Path) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    demands = sum(len(event.get("selected_expert_ids", event.get("selected", []))) for event in events)
    hits = sum(int(event["hits"]) for event in events)
    misses = sum(int(event.get("misses", event.get("blocking_misses", 0))) for event in events)
    if hits + misses != demands:
        raise ValueError(f"cache demands do not reconcile: {hits}+{misses}!={demands}")
    return {
        "events": len(events),
        "expert_demands": demands,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / demands if demands else None,
        "bytes_transferred": sum(int(event["bytes_transferred"]) for event in events),
        "blocking_wall_ms": sum(float(event["blocking_duration_us"]) for event in events) / 1000.0,
    }


def compare_modes(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    def change(key: str, higher_is_better: bool) -> float | None:
        before = baseline.get(key)
        after = candidate.get(key)
        if before in (None, 0) or after is None:
            return None
        raw = (float(after) - float(before)) / float(before) * 100.0
        return raw if higher_is_better else -raw

    return {
        "decode_tps_percent": change("decode_tps", True),
        "prompt_eval_tps_percent": change("prompt_eval_tps", True),
        "end_to_end_percent": change("end_to_end_ms", False),
    }
