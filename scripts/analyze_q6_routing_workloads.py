from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from expertflow.analysis.q6_routing import summarize_routing_locality
from expertflow.trace.io import load_router_events


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Q6 routing-workload locality")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    traces: dict[str, tuple] = {}
    families: dict[str, dict[str, tuple]] = {}
    for shard in manifest["shards"]:
        if shard.get("status") != "passed":
            continue
        events = tuple(load_router_events(Path(shard["trace"]["path"])))
        workload_id = str(shard["workload_id"])
        family = str(shard["workload_family"])
        traces[workload_id] = events
        families.setdefault(family, {})[workload_id] = events
    output = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured_routing_with_simulated_lru",
        "collection_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256(args.manifest),
            "shards": len(traces),
            "events": sum(len(events) for events in traces.values()),
        },
        "q6_expert_bundle_bytes": 5_358_852,
        "capacities": [112, 96, 80, 64],
        "predictor": {
            "status": "unavailable_for_q6",
            "reason": "Existing frozen predictors were trained and validated on the Q4 canonical runtime; no retraining or cross-quantization reinterpretation is allowed in this task."
        },
        "all_workloads": summarize_routing_locality(traces),
        "by_family": {
            family: summarize_routing_locality(rows)
            for family, rows in sorted(families.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["collection_manifest"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
