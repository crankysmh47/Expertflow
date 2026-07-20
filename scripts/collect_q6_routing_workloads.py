from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from expertflow.analysis.q6_routing import build_workloads, merged_workload_ids, probe_command
from expertflow.trace.io import load_router_events


MODEL_SHA256 = "089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba"
RUNTIME_SHA256 = "7ea12e0c44258bb2d75f99b7e180e7ef7cb0c7d3e285ca231fe26ad8c8c4932c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect bounded Q6 canonical routing workloads")
    parser.add_argument("--ppl-corpus", type=Path, required=True)
    parser.add_argument("--mmlu-manifest", type=Path, required=True)
    parser.add_argument("--conversation-corpus", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--max-workloads", type=int)
    args = parser.parse_args()

    probe = (args.runtime / "expertflow-router-probe.exe").resolve()
    model = args.model.resolve()
    if sha256(probe) != RUNTIME_SHA256:
        raise SystemExit("canonical observer runtime hash mismatch")
    if sha256(model) != MODEL_SHA256:
        raise SystemExit("Q6 model hash mismatch")
    mmlu = json.loads(args.mmlu_manifest.read_text(encoding="utf-8"))
    conversations = json.loads(args.conversation_corpus.read_text(encoding="utf-8"))
    workloads = list(
        build_workloads(
            ppl_text=args.ppl_corpus.read_text(encoding="utf-8"),
            mmlu_items=mmlu["mmlu"]["items"],
            conversations=conversations["conversations"],
        )
    )
    if args.families:
        selected = frozenset(args.families)
        workloads = [row for row in workloads if row.workload_family in selected]

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "collection-manifest.json"
    ledger_path = root / "command-ledger.jsonl"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "trace_generation": "trace_v2_canonical_segmented_q6",
            "cache_enabled": False,
            "predictor_enabled": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": artifact(model),
            "runtime": artifact(probe),
            "sources": {
                "ppl": artifact(args.ppl_corpus.resolve()),
                "mmlu": artifact(args.mmlu_manifest.resolve()),
                "conversations": artifact(args.conversation_corpus.resolve()),
            },
            "configuration": {"ngl": 10, "batch": 1, "ubatch": 1, "threads": 12, "sampler": "probe_default_greedy"},
            "workload_count": len(workloads),
            "shards": [],
        }
        write_json(manifest_path, manifest)
    manifest["workload_ids"] = list(merged_workload_ids(manifest["shards"], workloads))
    manifest["workload_count"] = len(manifest["workload_ids"])
    write_json(manifest_path, manifest)
    completed = {row["workload_id"] for row in manifest["shards"] if row.get("status") == "passed"}
    pending = [row for row in workloads if row.workload_id not in completed]
    if args.max_workloads is not None:
        pending = pending[: args.max_workloads]

    environment = {key: value for key, value in os.environ.items() if "EXPERTFLOW" not in key.upper()}
    environment["PATH"] = str(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")) + os.pathsep + str(args.runtime.resolve()) + os.pathsep + environment["PATH"]
    for workload in pending:
        shard = root / workload.workload_id
        shard.mkdir(parents=True, exist_ok=True)
        raw_trace = shard / "trace.raw.jsonl"
        trace = shard / "trace.jsonl"
        tokens = shard / "tokens.json"
        stdout = shard / "stdout.log"
        stderr = shard / "stderr.log"
        command = probe_command(
            workload,
            probe=str(probe),
            model=str(model),
            tokens=str(tokens),
            trace=str(raw_trace),
        )
        started = datetime.now(timezone.utc)
        clock = time.perf_counter()
        with stdout.open("wb") as output, stderr.open("wb") as errors:
            completed_run = subprocess.run(
                command,
                cwd=args.runtime.resolve(),
                env=environment,
                stdout=output,
                stderr=errors,
                check=False,
            )
        duration = time.perf_counter() - clock
        ledger = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workload_id": workload.workload_id,
            "family": workload.workload_family,
            "started_at": started.isoformat(),
            "duration_seconds": duration,
            "command": command,
            "return_code": completed_run.returncode,
        }
        with ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(ledger, separators=(",", ":")) + "\n")
        if completed_run.returncode != 0:
            raise SystemExit(f"{workload.workload_id} failed with {completed_run.returncode}")

        with raw_trace.open(encoding="utf-8") as source, trace.open("w", encoding="utf-8", newline="\n") as destination:
            for line in source:
                record = json.loads(line)
                record["request_id"] = f"request-{workload.workload_id}"
                record["conversation_id"] = workload.workload_id
                destination.write(json.dumps(record, separators=(",", ":")) + "\n")
        events = tuple(load_router_events(trace))
        layers = sorted({event.layer_id for event in events})
        if layers != list(range(30)) or any(len(event.selected_expert_ids) != 8 for event in events):
            raise SystemExit(f"{workload.workload_id} has incomplete routing coverage")
        saved = {
            **asdict(workload),
            "status": "passed",
            "duration_seconds": duration,
            "event_count": len(events),
            "forward_count": len({event.forward_id for event in events}),
            "layer_count": len(layers),
            "prompt_sha256": hashlib.sha256(workload.prompt.encode("utf-8")).hexdigest(),
            "trace": artifact(trace),
            "raw_trace": artifact(raw_trace),
            "tokens": artifact(tokens),
            "stdout": artifact(stdout),
            "stderr": artifact(stderr),
        }
        manifest["shards"].append(saved)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        print(f"passed {workload.workload_id}: {len(events)} events in {duration:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
