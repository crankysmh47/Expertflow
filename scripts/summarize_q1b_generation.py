from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


GENERATION_MODES = (
    "off",
    "layer0",
    "layer15",
    "layer20",
    "layers-0-15",
    "layers-0-1-15-20-r1",
    "layers-0-1-15-20-r2",
    "layers-0-1-15-20-r3",
)
PERFORMANCE_GROUPS = {
    "off": ("off-r1", "off-r2", "off-r3"),
    "layer0": ("layer0",),
    "layer15": ("layer15",),
    "layer20": ("layer20",),
    "layers-0-15": ("layers-0-15",),
    "layers-0-1-15-20": (
        "layers-0-1-15-20-r1",
        "layers-0-1-15-20-r2",
        "layers-0-1-15-20-r3",
    ),
}


def parse_perf_line(line: str) -> tuple[float, float]:
    match = re.search(r"Prompt:\s*([0-9.]+) t/s\s*\|\s*Generation:\s*([0-9.]+) t/s", line)
    if match is None:
        raise ValueError(f"invalid llama-cli performance line: {line!r}")
    return float(match.group(1)), float(match.group(2))


def load_trace(path: Path) -> dict[tuple[str, int, int], list[int]]:
    events: dict[tuple[str, int, int], list[int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            key = (str(event["phase"]), int(event["token_index"]), int(event["layer_id"]))
            if key in events:
                raise ValueError(f"duplicate routing event {key} in {path}")
            events[key] = [int(value) for value in event["selected_expert_ids"]]
    return events


def routing_overlap(
    reference: Mapping[tuple[str, int, int], Sequence[int]],
    candidate: Mapping[tuple[str, int, int], Sequence[int]],
) -> dict[str, Any]:
    common = sorted(set(reference) & set(candidate))
    set_matches = sum(set(reference[key]) == set(candidate[key]) for key in common)
    order_matches = sum(list(reference[key]) == list(candidate[key]) for key in common)
    denominator = len(reference)
    return {
        "reference_events": denominator,
        "candidate_events": len(candidate),
        "aligned_events": len(common),
        "set_matches": set_matches,
        "order_matches": order_matches,
        "set_overlap_rate": set_matches / denominator if denominator else 1.0,
        "order_overlap_rate": order_matches / denominator if denominator else 1.0,
    }


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "variance": statistics.variance(values) if len(values) > 1 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--performance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = sorted(path.name for path in (args.generation_root / "off" / "runs").iterdir() if path.is_dir())
    generation: dict[str, Any] = {}
    reference_tokens: dict[str, dict[str, Any]] = {}
    reference_traces: dict[str, dict[tuple[str, int, int], list[int]]] = {}
    for task in tasks:
        run = args.generation_root / "off" / "runs" / task / "observer"
        reference_tokens[task] = json.loads((run / "tokens.json").read_text(encoding="utf-8"))
        reference_traces[task] = load_trace(run / "trace.jsonl")

    for mode in GENERATION_MODES:
        mode_tasks: dict[str, Any] = {}
        durations: list[float] = []
        peak_gpu: list[int] = []
        for task in tasks:
            run = args.generation_root / mode / "runs" / task / "observer"
            measurement = json.loads((run / "measurement.json").read_text(encoding="utf-8"))
            tokens = json.loads((run / "tokens.json").read_text(encoding="utf-8"))
            trace = load_trace(run / "trace.jsonl")
            durations.append(float(measurement["duration_seconds"]))
            peak_gpu.append(int(measurement["memory"]["gpu_peak_used_mib"]["0"]))
            mode_tasks[task] = {
                "status": measurement["status"],
                "prompt_tokens_exact": tokens["prompt_token_ids"] == reference_tokens[task]["prompt_token_ids"],
                "generated_tokens_exact": tokens["generated_token_ids"] == reference_tokens[task]["generated_token_ids"],
                "routing": routing_overlap(reference_traces[task], trace),
            }
        generation[mode] = {
            "tasks": mode_tasks,
            "all_processes_passed": all(value["status"] == "passed" for value in mode_tasks.values()),
            "generated_token_agreement_rate": sum(value["generated_tokens_exact"] for value in mode_tasks.values()) / len(tasks),
            "router_set_overlap_rate": sum(value["routing"]["set_matches"] for value in mode_tasks.values())
            / sum(value["routing"]["reference_events"] for value in mode_tasks.values()),
            "router_order_overlap_rate": sum(value["routing"]["order_matches"] for value in mode_tasks.values())
            / sum(value["routing"]["reference_events"] for value in mode_tasks.values()),
            "duration_seconds": _summary(durations),
            "peak_system_gpu_used_mib": max(peak_gpu),
        }

    deterministic = True
    for task in tasks:
        baseline_mode = "layers-0-1-15-20-r1"
        baseline_run = args.generation_root / baseline_mode / "runs" / task / "observer"
        baseline_tokens = json.loads((baseline_run / "tokens.json").read_text(encoding="utf-8"))
        baseline_trace = load_trace(baseline_run / "trace.jsonl")
        for mode in ("layers-0-1-15-20-r2", "layers-0-1-15-20-r3"):
            run = args.generation_root / mode / "runs" / task / "observer"
            tokens = json.loads((run / "tokens.json").read_text(encoding="utf-8"))
            deterministic &= tokens["prompt_token_ids"] == baseline_tokens["prompt_token_ids"]
            deterministic &= tokens["generated_token_ids"] == baseline_tokens["generated_token_ids"]
            deterministic &= load_trace(run / "trace.jsonl") == baseline_trace

    performance: dict[str, Any] = {}
    for mode, repetitions in PERFORMANCE_GROUPS.items():
        prompt_values: list[float] = []
        decode_values: list[float] = []
        peak_values: list[int] = []
        for repetition in repetitions:
            run = args.performance_root / repetition
            measurement = json.loads((run / "measurement.json").read_text(encoding="utf-8"))
            stdout = (run / "stdout.log").read_text(encoding="utf-8", errors="replace")
            perf_line = next(line for line in stdout.splitlines() if "[ Prompt:" in line)
            prompt, decode = parse_perf_line(perf_line)
            prompt_values.append(prompt)
            decode_values.append(decode)
            peak_values.append(int(measurement["memory"]["gpu_peak_used_mib"]["0"]))
        performance[mode] = {
            "prompt_tps": _summary(prompt_values),
            "decode_tps": _summary(decode_values),
            "peak_system_gpu_used_mib": peak_values,
        }
    off_decode = performance["off"]["decode_tps"]["mean"]
    final_decode = performance["layers-0-1-15-20"]["decode_tps"]["mean"]
    performance["best4_decode_relative_change"] = final_decode / off_decode - 1.0

    result = {
        "generation": generation,
        "best4_three_repetition_determinism": deterministic,
        "performance": performance,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "best4_deterministic": deterministic,
        "best4_token_agreement": generation["layers-0-1-15-20-r1"]["generated_token_agreement_rate"],
        "best4_router_set_overlap": generation["layers-0-1-15-20-r1"]["router_set_overlap_rate"],
        "best4_router_order_overlap": generation["layers-0-1-15-20-r1"]["router_order_overlap_rate"],
        "best4_decode_relative_change": performance["best4_decode_relative_change"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
