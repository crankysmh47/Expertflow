from __future__ import annotations

import argparse
import json
from pathlib import Path

from expertflow.analysis.q6_routing import simulate_hybrid_candidate


SELECTED = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate bounded Q6 static/reactive hybrids")
    parser.add_argument("--locality", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locality = json.loads(args.locality.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    profile_by_layer = {int(row["layer"]): row for row in profile["layers"]}
    cache_order = sorted(SELECTED, key=lambda layer: (profile_by_layer[layer]["total_us_median"], layer))
    performance = locality["by_family"]["performance_512"]
    misses = {
        capacity: {layer: performance[str(layer)]["lru"][str(capacity)]["misses_per_token"] for layer in SELECTED}
        for capacity in (112, 96)
    }
    q4_effective_copy_mib_s = 4768.7
    q6_bundle_bytes = 5_358_852
    transfer_ms = q6_bundle_bytes / (1024 * 1024) / q4_effective_copy_mib_s * 1000.0
    intrinsic_fraction_per_layer = 0.0996 / 9.0
    candidates = []
    for capacity in (112, 96):
        for cached_count in (2, 4, 6):
            cached = tuple(cache_order[:cached_count])
            result = simulate_hybrid_candidate(
                static_tps=28.13,
                cached_layers=cached,
                capacity=capacity,
                full_slots=128,
                shadow_bytes_per_layer=685_933_056,
                misses_per_token=misses[capacity],
                transfer_ms_per_miss=transfer_ms,
                intrinsic_overhead_fraction_per_layer=intrinsic_fraction_per_layer,
            )
            result["static_layers"] = [layer for layer in SELECTED if layer not in cached]
            result["hit_rate_by_cached_layer"] = {
                str(layer): performance[str(layer)]["lru"][str(capacity)]["hit_rate"]
                for layer in cached
            }
            result["h2d_mib_per_token"] = result["misses_per_token"] * q6_bundle_bytes / (1024 * 1024)
            result["projected_peak_mib"] = 10966.801 - result["freed_mib"] + 16.0
            result["scheduler_metadata_reserve_mib"] = 16.0
            result["additional_full_layers_enabled"] = max(0, int((result["freed_mib"] - 16.0) // 654.15673828125))
            result["memory_gate_pass"] = result["freed_mib"] >= 500.0
            result["throughput_gate_pass"] = result["projected_tps"] >= 26.72
            result["candidate_gate_pass"] = result["memory_gate_pass"] and result["throughput_gate_pass"]
            candidates.append(result)

    selected_us = sum(profile_by_layer[layer]["total_us_median"] for layer in SELECTED)
    unselected = sorted(
        (layer for layer in profile_by_layer if layer not in SELECTED),
        key=lambda layer: (-profile_by_layer[layer]["total_us_median"], layer),
    )
    optimistic_next = unselected[0]
    optimistic_next_gain = (28.13 - 22.28) * profile_by_layer[optimistic_next]["total_us_median"] / selected_us
    for candidate in candidates:
        enabled = candidate["additional_full_layers_enabled"]
        candidate["optimistic_added_layer"] = optimistic_next if enabled else None
        candidate["optimistic_total_tps_with_one_added_layer"] = candidate["projected_tps"] + optimistic_next_gain if enabled else candidate["projected_tps"]
        candidate["beats_static_with_added_layer"] = candidate["optimistic_total_tps_with_one_added_layer"] > 28.13

    output = {
        "schema_version": "1.0.0",
        "measurement_kind": "simulation_from_measured_q6_routing_and_measured_q4_cache_cost",
        "static_champion": {"layers": list(SELECTED), "decode_tps": 28.13, "peak_mib": 10966.801},
        "assumptions": {
            "q6_bundle_bytes": q6_bundle_bytes,
            "q4_live_effective_copy_mib_s": q4_effective_copy_mib_s,
            "scaled_q6_transfer_ms_per_miss": transfer_ms,
            "q4_measured_intrinsic_overhead_fraction_nine_layers": 0.0996,
            "intrinsic_overhead_fraction_per_cached_layer": intrinsic_fraction_per_layer,
            "metadata_reserve_mib": 16.0,
            "note": "Q6 misses are measured from routing traces. Transfer and intrinsic-remap costs are conservative cross-quantization projections from the existing exact Q4 live cache, not Q6 runtime measurements."
        },
        "cache_order_lowest_profile_value_first": cache_order,
        "placement_classification": {
            "STATIC": [layer for layer in SELECTED if layer not in cache_order[:6]],
            "CACHEABLE": cache_order[:6],
            "CPU": sorted(layer for layer in profile_by_layer if layer not in SELECTED),
            "rule": "Keep the six highest measured CPU expert-time layers static; treat the six lower-value selected layers as cache candidates ordered by profile value. All unselected layers remain CPU because no simulated hybrid passed admission."
        },
        "candidates": candidates,
        "selected_runtime_candidates": [candidate for candidate in candidates if candidate["candidate_gate_pass"]][:3],
        "gate": {
            "minimum_freed_mib": 500.0,
            "minimum_retained_tps": 26.72,
            "any_candidate_pass": any(candidate["candidate_gate_pass"] for candidate in candidates),
            "verdict": "NO CACHE OPPORTUNITY" if not any(candidate["candidate_gate_pass"] for candidate in candidates) else "PROCEED"
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
