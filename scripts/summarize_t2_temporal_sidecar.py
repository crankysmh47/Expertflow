from __future__ import annotations

import argparse
import json
from pathlib import Path

from expertflow.benchmark.performance import summarize_repetitions
from scripts.run_p1_live_shadow_suite import FOCUSED


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    domains: dict[str, object] = {}
    all_reactive: list[dict[str, object]] = []
    all_t2: list[dict[str, object]] = []
    all_comparisons: list[dict[str, object]] = []
    for task_id in FOCUSED:
        pairs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((args.root / task_id).glob("rep-*/summary.json"))
            if "rep-00" not in str(path)
        ]
        reactive = [pair["reactive"] for pair in pairs]
        t2 = [pair["t2"] for pair in pairs]
        comparisons = [pair["comparison"] for pair in pairs]
        domains[task_id] = {
            "repetitions": len(pairs),
            "parity": (
                "exact"
                if all(
                    pair["token_parity"] == "exact"
                    and pair["router_parity"] == "exact"
                    for pair in pairs
                )
                else "failed"
            ),
            "reactive": summarize_repetitions(reactive),
            "t2": summarize_repetitions(t2),
            "comparison": summarize_repetitions(comparisons),
        }
        all_reactive.extend(reactive)
        all_t2.extend(t2)
        all_comparisons.extend(comparisons)

    result = {
        "measurement_kind": "live_runtime_measured",
        "model": "Gemma 4 26B A4B Q4_0",
        "ngl": 10,
        "reactive_slots": 32,
        "sidecar_slots": 2,
        "physical_slots": 34,
        "domains": domains,
        "overall": {
            "reactive": summarize_repetitions(all_reactive),
            "t2": summarize_repetitions(all_t2),
            "comparison": summarize_repetitions(all_comparisons),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
