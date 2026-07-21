from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def mode_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    decode = [float(row["decode_tps"]) for row in rows]
    prompt = [float(row["prompt_tps"]) for row in rows]
    wall = [float(row["wall_seconds"]) for row in rows]
    return {
        "decode_tps_mean": statistics.mean(decode),
        "decode_tps_median": statistics.median(decode),
        "decode_tps_sample_sd": sample_sd(decode),
        "prompt_tps_mean": statistics.mean(prompt),
        "wall_seconds_mean": statistics.mean(wall),
        "process_owned_peak_mib": max(float(row["process_owned_peak_mib"]) for row in rows),
    }


def summarize(
    rows: list[dict[str, Any]], *, bootstrap_samples: int = 100_000, seed: int = 20260719
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid")]
    off = [row for row in valid if row["mode"] == "off"]
    on = [row for row in valid if row["mode"] == "on"]
    pair_ids = sorted({int(row["pair"]) for row in valid})
    if len(off) != len(on) or len(off) != len(pair_ids):
        raise ValueError("incomplete matched pairs")
    paired: list[float] = []
    for pair_id in pair_ids:
        off_row = next(row for row in off if int(row["pair"]) == pair_id)
        on_row = next(row for row in on if int(row["pair"]) == pair_id)
        paired.append((float(on_row["decode_tps"]) / float(off_row["decode_tps"]) - 1.0) * 100.0)
    rng = random.Random(seed)
    resamples = [statistics.mean(rng.choice(paired) for _ in paired) for _ in range(bootstrap_samples)]
    return {
        "pair_count": len(pair_ids),
        "off": mode_summary(off),
        "on": mode_summary(on),
        "paired_improvement_pct_values": paired,
        "paired_improvement_pct_mean": statistics.mean(paired),
        "paired_improvement_pct_median": statistics.median(paired),
        "paired_improvement_pct_sample_sd": sample_sd(paired),
        "paired_bootstrap_95_pct": [percentile(resamples, 0.025), percentile(resamples, 0.975)],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8-sig"))
    rows = list(raw["rows"] if isinstance(raw, dict) else raw)
    summary = summarize(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        fields = ["pair", "order", "mode", "decode_tps", "prompt_tps", "wall_seconds",
                  "process_owned_peak_mib", "response_sha256", "valid"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
