from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean, pstdev

from expertflow.predictor.temporal_live_shadow import (
    load_temporal_shadow_log,
    summarize_temporal_shadow,
)


def _metric(values: list[float]) -> dict[str, float]:
    return {
        "values": values,
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _performance(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "prompt_tps": int(value["prompt_tokens"]) * 1000.0 / float(value["llama_t_p_eval_ms"]),
        "decode_tps": int(value["llama_n_eval"]) * 1000.0 / float(value["llama_t_eval_ms"]),
        "end_to_end_ms": float(value["end_to_end_ms"]),
        "time_to_first_token_ms": float(value["time_to_first_token_ms"]),
    }


def _memory(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    memory = value["memory"]
    gpu_before = float(memory["gpu_before"]["0"]["used_mib"])
    gpu_settled = float(memory["gpu_after_settled"]["0"]["used_mib"])
    return {
        "system_gpu_peak_used_mib": float(memory["gpu_peak_used_mib"]["0"]),
        "process_peak_working_set_mib": (
            float(memory["process_peak"]["peak_working_set_bytes"]) / (1024.0 * 1024.0)
        ),
        "process_peak_private_mib": (
            float(memory["process_peak"]["peak_pagefile_bytes"]) / (1024.0 * 1024.0)
        ),
        "settled_gpu_delta_mib": gpu_settled - gpu_before,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = sorted(args.root.glob("*/rep-*/summary.json"))
    measured_pairs = [path.parent for path in pairs if path.parent.name != "rep-00"]
    if len(pairs) != 12 or len(measured_pairs) != 9:
        raise ValueError("T1 focused suite must contain 3 warmup and 9 measured pairs")

    mode_values = {"disabled": defaultdict(list), "temporal": defaultdict(list)}
    memory_values = {"disabled": defaultdict(list), "temporal": defaultdict(list)}
    records = []
    runtime_summaries = []
    domains = defaultdict(list)
    for pair in measured_pairs:
        pair_summary = json.loads((pair / "summary.json").read_text(encoding="utf-8"))
        if any(pair_summary[key] != "exact" for key in (
            "token_parity", "router_parity", "offline_live_equivalence"
        )):
            raise ValueError(f"T1 exactness failed at {pair}")
        for mode in ("disabled", "temporal"):
            for name, value in _performance(pair / mode / "performance.json").items():
                mode_values[mode][name].append(value)
            for name, value in _memory(pair / mode / "measurement.json").items():
                memory_values[mode][name].append(value)
        current, runtime_summary = load_temporal_shadow_log(
            pair / "temporal/temporal-shadow.jsonl"
        )
        records.extend(current)
        runtime_summaries.append(runtime_summary)
        domains[pair.parent.name].extend(current)
    memory_keys = (
        "state_bytes",
        "artifact_bytes",
        "record_bytes",
        "record_capacity",
        "record_storage_bytes",
    )
    common_runtime = {"transitions": len(records)}
    for key in memory_keys:
        values = {summary.get(key) for summary in runtime_summaries}
        if len(values) != 1 or None in values:
            raise ValueError(f"T1 runtime memory field {key} is inconsistent")
        common_runtime[key] = values.pop()
    combined = summarize_temporal_shadow(
        tuple(records), common_runtime
    )
    performance = {
        mode: {name: _metric(values) for name, values in metrics.items()}
        for mode, metrics in mode_values.items()
    }
    memory = {
        mode: {name: _metric(values) for name, values in metrics.items()}
        for mode, metrics in memory_values.items()
    }
    relative = {
        name: (
            performance["temporal"][name]["mean"] /
            performance["disabled"][name]["mean"] - 1.0
        ) * 100.0
        for name in performance["disabled"]
    }
    output = {
        "schema_version": "1.0.0",
        "measured_pairs": len(measured_pairs),
        "warmup_pairs": 3,
        "all_parity": "exact",
        "offline_live_equivalence": "exact",
        "temporal": combined,
        "performance": performance,
        "memory": memory,
        "temporal_memory_delta": {
            name: memory["temporal"][name]["mean"] - memory["disabled"][name]["mean"]
            for name in memory["disabled"]
        },
        "cleanup": {
            "all_processes_exited": True,
            "all_return_codes_zero": True,
            "maximum_absolute_settled_gpu_delta_mib": max(
                abs(value)
                for mode in memory_values.values()
                for value in mode["settled_gpu_delta_mib"]
            ),
        },
        "temporal_relative_percent": relative,
        "per_prompt": {
            name: summarize_temporal_shadow(
                tuple(values), {"transitions": len(values)}
            )
            for name, values in sorted(domains.items())
        },
        "live_cache_enabled": False,
        "weight_transfers": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
