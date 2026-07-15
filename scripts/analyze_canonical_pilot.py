from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from expertflow.trace.io import load_router_events


def hit_rate(hit_count: int, demand_count: int) -> dict[str, object]:
    return {"hit_count": hit_count, "demand_count": demand_count, "hit_rate": hit_count / demand_count}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--heldout-curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    curve = json.loads(args.heldout_curve.read_text(encoding="utf-8"))
    shards = manifest["shards"]
    evaluation_groups = []
    for shard in shards:
        if shard["split"] in {"validation", "test"}:
            events = [event for event in load_router_events(Path(shard["trace"]["path"])) if event.phase == "decode"]
            evaluation_groups.append((shard, events))

    adjacent_hits = 0
    adjacent_demands = 0
    for _, events in evaluation_groups:
        previous = {}
        for event in events:
            selected = set(event.selected_expert_ids)
            if event.layer_id in previous:
                adjacent_hits += len(selected & previous[event.layer_id])
                adjacent_demands += len(selected)
            previous[event.layer_id] = selected

    points = []
    for curve_point in curve["points"]:
        capacity = curve_point["capacity_per_layer"]
        session_hits = 0
        oracle_hits = 0
        demands = 0
        for _, events in evaluation_groups:
            seen = defaultdict(Counter)
            final = defaultdict(Counter)
            for event in events:
                final[event.layer_id].update(event.selected_expert_ids)
            oracle_residents = {
                layer: {expert for expert, _ in counts.most_common(capacity)}
                for layer, counts in final.items()
            }
            for event in events:
                residents = {expert for expert, _ in seen[event.layer_id].most_common(capacity)}
                session_hits += sum(expert in residents for expert in event.selected_expert_ids)
                oracle_hits += sum(expert in oracle_residents[event.layer_id] for expert in event.selected_expert_ids)
                demands += len(event.selected_expert_ids)
                seen[event.layer_id].update(event.selected_expert_ids)
        points.append({
            "capacity_per_layer": capacity,
            "training_static": curve_point["static_hotset"],
            "conversation_reset_lru": curve_point["lru"],
            "session_frequency": hit_rate(session_hits, demands),
            "hindsight_session_static_upper_bound": hit_rate(oracle_hits, demands),
        })
    report = {
        "schema_version": "1.0.0", "measurement_kind": "measured_routing_estimated_policy",
        "trace_generation": "trace_v2_canonical_segmented",
        "conversation_separation": "static fit uses train only; all reported scores use validation/test only",
        "training_conversation_count": sum(shard["split"] == "train" for shard in shards),
        "evaluation_conversation_count": len(evaluation_groups),
        "adjacent_token_reuse": hit_rate(adjacent_hits, adjacent_demands),
        "points": points,
        "oracle_note": "Hindsight upper bound for a fixed per-conversation resident set; not a live or deadline result.",
        "live_cache_enabled": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
