from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
from typing import Any


_LAYER = re.compile(r"^ffn_moe_gate_up-(\d+)$")


def rank_profiles(
    paths: list[Path], *, selected_layers: set[int], shadow_bytes: int
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one profile is required")
    samples: dict[int, list[dict[str, Any]]] = {}
    profile_totals: list[int] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        records = list(data["records"])
        profile_totals.append(sum(int(record["total_us"]) for record in records))
        seen: set[int] = set()
        for record in records:
            match = _LAYER.match(str(record.get("first_node", "")))
            if not match:
                continue
            layer = int(match.group(1))
            if layer in seen:
                raise ValueError(f"duplicate expert split for layer {layer} in {path}")
            seen.add(layer)
            samples.setdefault(layer, []).append(record)
    if any(len(values) != len(paths) for values in samples.values()):
        raise ValueError("profile layer sets differ")
    shadow_mib = shadow_bytes / (1024 * 1024)
    total_median = statistics.median(profile_totals)
    layers: list[dict[str, Any]] = []
    for layer, values in samples.items():
        total_us = statistics.median(float(value["total_us"]) for value in values)
        compute_us = statistics.median(float(value["compute_submit_us"]) for value in values)
        input_us = statistics.median(float(value["input_boundary_us"]) for value in values)
        completion_us = statistics.median(float(value["completion_us"]) for value in values)
        penalty = 1.0
        layers.append({
            "layer": layer,
            "selected_initially": layer in selected_layers,
            "backend": str(values[0].get("backend", "unknown")),
            "total_us_samples": [int(value["total_us"]) for value in values],
            "total_us_median": total_us,
            "compute_submit_us_median": compute_us,
            "input_boundary_us_median": input_us,
            "completion_us_median": completion_us,
            "profile_total_share_pct": total_us / total_median * 100.0,
            "shadow_bytes": shadow_bytes,
            "shadow_mib": shadow_mib,
            "quality_risk_penalty": penalty,
            "score_us_per_mib": total_us / shadow_mib * penalty,
        })
    layers.sort(key=lambda item: (-float(item["score_us_per_mib"]), int(item["layer"])))
    return {
        "schema_version": "1.0.0",
        "measurement_kind": "repeated_synchronized_split_profile_ranking",
        "diagnostic_only": True,
        "profile_count": len(paths),
        "profile_paths": [str(path.resolve()) for path in paths],
        "profile_total_us_samples": profile_totals,
        "profile_total_us_median": total_median,
        "selected_initial_layers": sorted(selected_layers),
        "ranking_formula": "median selected CPU expert split total_us / shadow_mib * quality_risk_penalty",
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank Q6 routed layers by profiled CPU time per shadow MiB")
    parser.add_argument("--profile", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected", default="0,1,15,20")
    parser.add_argument("--shadow-bytes", type=int, default=685_933_056)
    args = parser.parse_args()
    selected = {int(value) for value in args.selected.split(",") if value}
    result = rank_profiles(args.profile, selected_layers=selected, shadow_bytes=args.shadow_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
