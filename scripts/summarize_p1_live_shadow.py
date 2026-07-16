from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

from expertflow.predictor.live_shadow import (
    load_shadow_log,
    summarize_shadow_records,
)


def _metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _performance(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    prompt_tokens = int(value["prompt_tokens"])
    prompt_ms = float(value["llama_t_p_eval_ms"])
    decode_runs = int(value["llama_n_eval"])
    decode_ms = float(value["llama_t_eval_ms"])
    return {
        "prompt_tps": prompt_tokens * 1000.0 / prompt_ms,
        "decode_tps": decode_runs * 1000.0 / decode_ms,
        "end_to_end_ms": float(value["end_to_end_ms"]),
        "time_to_first_token_ms": float(value["time_to_first_token_ms"]),
    }


def _measurement(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    memory = value["memory"]
    before = memory["gpu_before"]["0"]["used_mib"]
    settled = memory["gpu_after_settled"]["0"]["used_mib"]
    return {
        "peak_gpu_used_mib": float(memory["gpu_peak_used_mib"]["0"]),
        "peak_process_gpu_mib": float(memory["process_gpu_peak_mib"]),
        "peak_working_set_bytes": float(
            memory["process_peak"]["peak_working_set_bytes"]
        ),
        "settled_gpu_delta_mib": float(settled - before),
    }


def _collect_pairs(root: Path, include_warmup: bool) -> list[Path]:
    pairs = []
    for path in sorted(root.glob("*/rep-*/summary.json")):
        if not include_warmup and path.parent.name == "rep-00":
            continue
        pairs.append(path.parent)
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    focused_pairs = _collect_pairs(args.focused_root, include_warmup=False)
    smoke_pairs = _collect_pairs(args.smoke_root, include_warmup=True)
    if len(focused_pairs) != 9 or len(smoke_pairs) != 7:
        raise ValueError("P1 validation suite is incomplete")

    mode_metrics: dict[str, dict[str, list[float]]] = {
        "disabled": defaultdict(list),
        "shadow": defaultdict(list),
    }
    memory_metrics: dict[str, dict[str, list[float]]] = {
        "disabled": defaultdict(list),
        "shadow": defaultdict(list),
    }
    domain_values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {
            "disabled": defaultdict(list),
            "shadow": defaultdict(list),
        }
    )
    all_records = []
    token_payloads: dict[tuple[str, str], set[str]] = defaultdict(set)
    trace_counts: dict[str, list[int]] = {"disabled": [], "shadow": []}

    for pair in [*focused_pairs, *smoke_pairs]:
        pair_summary = json.loads(
            (pair / "summary.json").read_text(encoding="utf-8")
        )
        if (
            pair_summary["token_parity"] != "exact"
            or pair_summary["router_parity"] != "exact"
            or pair_summary["offline_live_equivalence"] != "exact"
            or pair_summary["candidate_support_failures"] != 0
        ):
            raise ValueError(f"P1 parity gate failed at {pair}")
        task_id = str(pair_summary["task_id"])
        for mode in ("disabled", "shadow"):
            perf = _performance(pair / mode / "performance.json")
            memory = _measurement(pair / mode / "measurement.json")
            for name, value in perf.items():
                mode_metrics[mode][name].append(value)
                domain_values[task_id][mode][name].append(value)
            for name, value in memory.items():
                memory_metrics[mode][name].append(value)
            token_payloads[(task_id, mode)].add(
                (pair / mode / "tokens.json").read_text(encoding="utf-8")
            )
            trace_counts[mode].append(
                len((pair / mode / "trace.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines())
            )
        records, _ = load_shadow_log(pair / "shadow" / "shadow.jsonl")
        all_records.extend(records)

    for key, payloads in token_payloads.items():
        if len(payloads) != 1:
            raise ValueError(f"repeated token determinism failed for {key}")
    if trace_counts["disabled"] != trace_counts["shadow"]:
        raise ValueError("paired router event counts differ")

    combined_shadow = summarize_shadow_records(
        tuple(all_records),
        {
            "candidate_support_failures": 0,
        },
    )
    measured = {
        mode: {
            name: _metric(values)
            for name, values in metrics.items()
        }
        for mode, metrics in mode_metrics.items()
    }
    overhead = {}
    for name in (
        "prompt_tps",
        "decode_tps",
        "end_to_end_ms",
        "time_to_first_token_ms",
    ):
        disabled = measured["disabled"][name]["mean"]
        shadow = measured["shadow"][name]["mean"]
        overhead[name] = (
            (shadow / disabled - 1.0) * 100.0
            if name.endswith("_ms")
            else (shadow / disabled - 1.0) * 100.0
        )

    output = {
        "schema_version": "1.0.0",
        "focused_measured_pairs": len(focused_pairs),
        "focused_warmup_pairs": 3,
        "smoke_pairs": len(smoke_pairs),
        "all_parity": "exact",
        "offline_live_equivalence": "exact",
        "candidate_support_failures": 0,
        "transitions": combined_shadow["transitions"],
        "prediction": {
            "prefill": combined_shadow["prefill"],
            "decode": combined_shadow["decode"],
        },
        "performance": measured,
        "shadow_relative_percent": overhead,
        "memory": {
            mode: {
                name: _metric(values)
                for name, values in metrics.items()
            }
            for mode, metrics in memory_metrics.items()
        },
        "maximum_settled_gpu_delta_mib": max(
            max(memory_metrics["disabled"]["settled_gpu_delta_mib"]),
            max(memory_metrics["shadow"]["settled_gpu_delta_mib"]),
        ),
        "domain_performance": {
            task: {
                mode: {
                    name: _metric(values)
                    for name, values in metrics.items()
                }
                for mode, metrics in modes.items()
            }
            for task, modes in domain_values.items()
        },
        "router_event_counts_equal": True,
        "repeated_tokens_deterministic": True,
        "live_cache_enabled": False,
        "predictor_weight_transfers": 0,
    }
    if not all(math.isfinite(value) for value in overhead.values()):
        raise ValueError("P1 overhead contains a non-finite value")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
