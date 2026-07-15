from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from expertflow.expanded_collection import (
    load_expanded_manifest,
    select_collection_rows,
    validate_canonical_shard,
    validate_selection_lock,
)


EXPECTED_RUNTIME_SHA256 = "7ea12e0c44258bb2d75f99b7e180e7ef7cb0c7d3e285ca231fe26ad8c8c4932c"
EXPECTED_MODEL_SHA256 = "4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5"
PROMPT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def artifact_valid(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    path = Path(str(record.get("path", "")))
    return (
        path.is_file()
        and record.get("bytes") == path.stat().st_size
        and record.get("sha256") == sha256(path)
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_ledger(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--measurement-runner", type=Path, required=True)
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--unseal-test", action="store_true")
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--selection-commit")
    args = parser.parse_args()

    corpus = load_expanded_manifest(args.manifest.resolve())
    selected = list(
        select_collection_rows(
            corpus, splits=tuple(args.splits), unseal_test=args.unseal_test
        )
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    probe = args.runtime.resolve() / "expertflow-router-probe.exe"
    model = args.model.resolve()
    runner = args.measurement_runner.resolve()
    runtime_hash = sha256(probe)
    model_hash = sha256(model)
    if runtime_hash != EXPECTED_RUNTIME_SHA256:
        raise SystemExit(f"canonical runtime hash mismatch: {runtime_hash}")
    if model_hash != EXPECTED_MODEL_SHA256:
        raise SystemExit(f"canonical model hash mismatch: {model_hash}")

    manifest_path = root / "collection-manifest.json"
    ledger_path = root / "command-ledger.jsonl"
    if manifest_path.exists():
        collection = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        collection = {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "trace_generation": "trace_v2_canonical_segmented",
            "canonical_runtime": "expertflow-canonical-observer-v1",
            "cache_enabled": False,
            "test_split_sealed": not args.unseal_test,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frozen_corpus": artifact(args.manifest.resolve()),
            "runtime": artifact(probe),
            "model": artifact(model),
            "measurement_runner": artifact(runner),
            "configuration": {
                "n_predict": 32,
                "gpu_layers": 10,
                "threads": 12,
                "batch": 1,
                "microbatch": 1,
                "sampler": "greedy",
                "prompt_template": PROMPT_TEMPLATE,
            },
            "shards": [],
        }
        write_json(manifest_path, collection)

    if args.unseal_test:
        if args.selection_lock is None or args.selection_commit != "2300c6c":
            raise SystemExit("expanded test unseal requires pinned selection lock and commit 2300c6c")
        collection["test_split_sealed"] = False
        collection["test_unseal_provenance"] = {
            "predictor_commit": args.selection_commit,
            "selection_lock": validate_selection_lock(args.selection_lock.resolve()),
            "train_validation_checkpoint": {
                "train": 60,
                "validation": 12,
                "artifact_revalidation": "passed before unseal",
            },
            "unsealed_at": datetime.now(timezone.utc).isoformat(),
            "scope": "collection only; no expanded-test training or evaluation",
        }
        write_json(manifest_path, collection)

    completed_ids = {
        shard["conversation_id"]
        for shard in collection["shards"]
        if shard.get("status") == "passed"
        and all(
            artifact_valid(shard.get(name))
            for name in ("trace", "raw_trace", "tokens", "measurement")
        )
    }
    pending = [row for row in selected if row.conversation_id not in completed_ids]
    if args.max_conversations is not None:
        pending = pending[: args.max_conversations]

    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"))
        + os.pathsep
        + environment["PATH"]
    )
    for row in pending:
        shard_dir = root / row.conversation_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        raw_trace = shard_dir / "trace.raw.jsonl"
        trace = shard_dir / "trace.jsonl"
        tokens = shard_dir / "tokens.json"
        measurement = shard_dir / "measurement.json"
        command = [
            str(probe), "-m", str(model), "--tokens", str(tokens),
            "--trace", str(raw_trace), "--trace-mode", "full",
            "-n", "32", "-ngl", "10", "--threads", "12",
            PROMPT_TEMPLATE.format(prompt=row.prompt),
        ]
        measured = [
            sys.executable, str(runner), "--manifest", str(measurement),
            "--cwd", str(args.runtime.resolve()), "--settle-seconds", "2",
            "--artifact", f"tokens={tokens}", "--artifact", f"raw_trace={raw_trace}",
            "--", *command,
        ]
        started_at = datetime.now(timezone.utc)
        started_ns = time.perf_counter_ns()
        completed = subprocess.run(measured, env=environment, check=False)
        duration = (time.perf_counter_ns() - started_ns) / 1e9
        ledger_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversation_id": row.conversation_id,
            "split": row.split,
            "domain": row.domain,
            "command": measured,
            "cwd": str(args.runtime.resolve()),
            "started_at": started_at.isoformat(),
            "duration_seconds": duration,
            "return_code": completed.returncode,
        }
        if completed.returncode != 0:
            ledger_record["result"] = "failed_native_process"
            append_ledger(ledger_path, ledger_record)
            return completed.returncode

        event_count = 0
        with raw_trace.open(encoding="utf-8") as source, trace.open(
            "w", encoding="utf-8", newline="\n"
        ) as output:
            for line in source:
                event = json.loads(line)
                event["request_id"] = f"request-{row.conversation_id}"
                event["conversation_id"] = row.conversation_id
                output.write(json.dumps(event, separators=(",", ":")) + "\n")
                event_count += 1
        validation = validate_canonical_shard(trace, row.conversation_id)
        if event_count != validation["event_count"]:
            raise SystemExit("canonicalization event count mismatch")
        shard = {
            "conversation_id": row.conversation_id,
            "split": row.split,
            "domain": row.domain,
            "template_id": row.template_id,
            "prompt_sha256": row.prompt_sha256,
            "status": "passed",
            "validation": validation,
            "trace": artifact(trace),
            "raw_trace": artifact(raw_trace),
            "tokens": artifact(tokens),
            "measurement": artifact(measurement),
        }
        collection["shards"].append(shard)
        collection["updated_at"] = datetime.now(timezone.utc).isoformat()
        counts = {split: 0 for split in ("train", "validation", "test")}
        for saved in collection["shards"]:
            if saved.get("status") == "passed":
                counts[saved["split"]] += 1
        collection["completed_counts"] = counts
        write_json(manifest_path, collection)
        ledger_record.update({"result": "passed", "validation": validation})
        append_ledger(ledger_path, ledger_record)
        print(f"passed {row.conversation_id}: {event_count} events in {duration:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
