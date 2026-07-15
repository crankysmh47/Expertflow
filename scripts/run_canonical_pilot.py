from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SELECTED_IDS = (
    "train-general-01", "train-code-01", "train-math-01",
    "train-translation-01", "train-multilingual-01",
    "train-structured-01", "train-shift-01",
    "validation-general-10", "validation-code-08", "validation-math-06",
    "validation-translation-04", "test-multilingual-04",
    "test-structured-02", "test-shift-02",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--measurement-runner", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source_corpus.read_text(encoding="utf-8"))
    by_id = {row["conversation_id"]: row for row in source["conversations"]}
    conversations = [by_id[identifier] for identifier in SELECTED_IDS]
    frozen = {
        "schema_version": "1.0.0", "dataset_id": "trace_v2_canonical_segmented_pilot",
        "source_policy": source["source_policy"],
        "split_policy": "Complete conversations selected and frozen before canonical trace generation.",
        "conversation_ids": list(SELECTED_IDS), "conversations": conversations,
    }
    frozen_path = root / "pilot-corpus.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = root / "collection-manifest.json"
    manifest = {
        "schema_version": "1.0.0", "measurement_kind": "measured",
        "trace_generation": "trace_v2_canonical_segmented",
        "canonical_runtime": "expertflow-canonical-observer-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": artifact(frozen_path), "shards": [],
    }
    probe = args.runtime.resolve() / "expertflow-router-probe.exe"
    env = os.environ.copy()
    env["PATH"] = str(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")) + os.pathsep + env["PATH"]
    template = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    for row in conversations:
        shard = root / row["conversation_id"]
        shard.mkdir(parents=True, exist_ok=True)
        raw_trace = shard / "trace.raw.jsonl"
        canonical_trace = shard / "trace.jsonl"
        tokens = shard / "tokens.json"
        measurement = shard / "measurement.json"
        command = [
            str(probe), "-m", str(args.model.resolve()), "--tokens", str(tokens),
            "--trace", str(raw_trace), "--trace-mode", "full", "-n", "32",
            "-ngl", "10", "--threads", "12", template.format(prompt=row["prompt"]),
        ]
        measured = [
            sys.executable, str(args.measurement_runner.resolve()), "--manifest", str(measurement),
            "--cwd", str(args.runtime.resolve()), "--settle-seconds", "2",
            "--artifact", f"tokens={tokens}", "--artifact", f"raw_trace={raw_trace}",
            "--", *command,
        ]
        completed = subprocess.run(measured, env=env, check=False)
        if completed.returncode != 0:
            return completed.returncode
        event_count = 0
        with raw_trace.open(encoding="utf-8") as input_file, canonical_trace.open("w", encoding="utf-8", newline="\n") as output_file:
            for line in input_file:
                event = json.loads(line)
                event["request_id"] = f"request-{row['conversation_id']}"
                event["conversation_id"] = row["conversation_id"]
                output_file.write(json.dumps(event, separators=(",", ":")) + "\n")
                event_count += 1
        manifest["shards"].append({
            "conversation_id": row["conversation_id"], "split": row["split"],
            "domain": row["domain"], "message_count": row["message_count"],
            "status": "passed", "event_count": event_count,
            "trace": artifact(canonical_trace), "raw_trace": artifact(raw_trace),
            "tokens": artifact(tokens), "measurement": artifact(measurement),
        })
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["summary"] = {"conversation_count": len(conversations), "passed": len(conversations), "failed": 0}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
