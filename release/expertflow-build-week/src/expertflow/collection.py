"""Validated corpus contracts for reproducible router-trace collection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from expertflow.trace.io import load_router_events
from expertflow.trace.parity import compare_token_sequences


EXPECTED_SPLIT_COUNTS = {
    "train": 32,
    "validation": 4,
    "test": 4,
}

EXPECTED_DOMAIN_COUNTS = {
    "general_chat": 10,
    "code": 8,
    "math_reasoning": 6,
    "translation": 4,
    "multilingual": 4,
    "long_context": 4,
    "structured_output": 2,
    "topic_shift": 2,
}


@dataclass(frozen=True)
class CorpusConversation:
    """One independently split synthetic conversation."""

    conversation_id: str
    split: str
    domain: str
    source_dataset: str
    source_record_id: str
    message_count: int
    prompt: str


@dataclass(frozen=True)
class CorpusManifest:
    """Frozen physical-feasibility corpus and declared split contract."""

    schema_version: str
    dataset_id: str
    split_counts: dict[str, int]
    domain_counts: dict[str, int]
    conversations: tuple[CorpusConversation, ...]


@dataclass(frozen=True)
class CollectionConfig:
    """Pinned process and output settings for paired probe collection."""

    probe: Path
    model: Path
    model_sha256: str
    output_dir: Path
    n_predict: int = 64
    gpu_layers: int = 10
    threads: int = 12


@dataclass(frozen=True)
class NativeProcessResult:
    """Auditable native-process timing returned by the process boundary."""

    return_code: int
    started_at: str
    ended_at: str
    duration_seconds: float


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_count_map(value: Any, label: str) -> dict[str, int]:
    mapping = _require_object(value, label)
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in mapping.values()):
        raise ValueError(f"{label} values must be non-negative integers")
    return dict(mapping)


def load_corpus_manifest(
    path: Path,
    *,
    expected_split_counts: dict[str, int] | None = None,
    expected_domain_counts: dict[str, int] | None = None,
) -> CorpusManifest:
    """Load and strictly validate the frozen 40-conversation corpus."""

    required_splits = expected_split_counts or EXPECTED_SPLIT_COUNTS
    required_domains = expected_domain_counts or EXPECTED_DOMAIN_COUNTS

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read corpus manifest: {error}") from error
    root = _require_object(payload, "corpus manifest")
    schema_version = _require_string(root.get("schema_version"), "schema_version")
    if schema_version != "1.0.0":
        raise ValueError("unsupported corpus schema_version")
    dataset_id = _require_string(root.get("dataset_id"), "dataset_id")
    split_counts = _require_count_map(root.get("split_counts"), "split_counts")
    domain_counts = _require_count_map(root.get("domain_counts"), "domain_counts")
    if split_counts != required_splits:
        raise ValueError(f"split_counts must equal {required_splits}")
    if domain_counts != required_domains:
        raise ValueError(f"domain_counts must equal {required_domains}")

    rows = root.get("conversations")
    if not isinstance(rows, list):
        raise ValueError("conversations must be an array")
    conversations: list[CorpusConversation] = []
    for index, raw_row in enumerate(rows):
        row = _require_object(raw_row, f"conversations[{index}]")
        message_count = row.get("message_count")
        if isinstance(message_count, bool) or not isinstance(message_count, int) or message_count <= 0:
            raise ValueError(
                f"conversations[{index}].message_count must be a positive integer"
            )
        conversations.append(
            CorpusConversation(
                conversation_id=_require_string(
                    row.get("conversation_id"),
                    f"conversations[{index}].conversation_id",
                ),
                split=_require_string(
                    row.get("split"), f"conversations[{index}].split"
                ),
                domain=_require_string(
                    row.get("domain"), f"conversations[{index}].domain"
                ),
                source_dataset=_require_string(
                    row.get("source_dataset"),
                    f"conversations[{index}].source_dataset",
                ),
                source_record_id=_require_string(
                    row.get("source_record_id"),
                    f"conversations[{index}].source_record_id",
                ),
                message_count=message_count,
                prompt=_require_string(
                    row.get("prompt"), f"conversations[{index}].prompt"
                ),
            )
        )

    identifiers = [item.conversation_id for item in conversations]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("conversation_id values must be unique")
    source_ids = [item.source_record_id for item in conversations]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_record_id values must be unique")

    observed_splits = dict(Counter(item.split for item in conversations))
    if observed_splits != split_counts:
        raise ValueError(
            f"observed split counts {observed_splits} do not match {split_counts}"
        )
    observed_domains = dict(Counter(item.domain for item in conversations))
    if observed_domains != domain_counts:
        raise ValueError(
            f"observed domain counts {observed_domains} do not match {domain_counts}"
        )
    held_out_domains = {
        item.domain
        for item in conversations
        if item.split in {"validation", "test"}
    }
    requires_held_out_coverage = any(
        required_splits.get(split, 0) > 0 for split in ("validation", "test")
    )
    if requires_held_out_coverage and held_out_domains != set(required_domains):
        raise ValueError("validation/test rows must cover every declared domain")

    return CorpusManifest(
        schema_version=schema_version,
        dataset_id=dataset_id,
        split_counts=split_counts,
        domain_counts=domain_counts,
        conversations=tuple(conversations),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _artifact_is_valid(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    path_value = record.get("path")
    size = record.get("bytes")
    digest = record.get("sha256")
    if (
        not isinstance(path_value, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not isinstance(digest, str)
    ):
        return False
    path = Path(path_value)
    return path.is_file() and path.stat().st_size == size and _sha256(path) == digest


def _run_native_process(
    argv: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> NativeProcessResult:
    started = datetime.now(timezone.utc)
    started_ns = time.perf_counter_ns()
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            check=False,
            creationflags=creation_flags,
        )
    ended = datetime.now(timezone.utc)
    return NativeProcessResult(
        return_code=completed.returncode,
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
        duration_seconds=(time.perf_counter_ns() - started_ns) / 1_000_000_000,
    )


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _process_record(
    argv: list[str],
    result: NativeProcessResult,
    *,
    stdout_path: Path,
    stderr_path: Path,
    tokens_path: Path,
) -> dict[str, object]:
    return {
        "argv": argv,
        "return_code": result.return_code,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "duration_seconds": result.duration_seconds,
        "stdout": _artifact(stdout_path),
        "stderr": _artifact(stderr_path),
        "tokens": _artifact(tokens_path) if tokens_path.is_file() else None,
    }


def _attempt_is_valid(shard: dict[str, object]) -> bool:
    if shard.get("latest_status") != "passed":
        return False
    attempts = shard.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    attempt = attempts[-1]
    if not isinstance(attempt, dict):
        return False
    for run_name in ("baseline", "instrumented"):
        run = attempt.get(run_name)
        if not isinstance(run, dict) or run.get("return_code") != 0:
            return False
        if not all(
            _artifact_is_valid(run.get(field))
            for field in ("stdout", "stderr", "tokens")
        ):
            return False
    trace = attempt.get("trace")
    parity = attempt.get("parity")
    if not isinstance(trace, dict) or not isinstance(parity, dict):
        return False
    if not _artifact_is_valid(trace.get("artifact")):
        return False
    if not _artifact_is_valid(parity.get("artifact")):
        return False
    if not (parity.get("prompt_matches") and parity.get("generated_matches")):
        return False
    try:
        event_count = sum(
            1 for _ in load_router_events(Path(trace["artifact"]["path"]))
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return event_count == trace.get("event_count") and event_count > 0


def _identity(
    corpus_path: Path,
    corpus: CorpusManifest,
    config: CollectionConfig,
) -> dict[str, object]:
    probe = config.probe.resolve()
    model = config.model.resolve()
    runtime_files = sorted(probe.parent.glob("*.dll"))
    return {
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": _sha256(corpus_path.resolve()),
        "dataset_id": corpus.dataset_id,
        "probe": _artifact(probe),
        "runtime_dlls": [_artifact(path) for path in runtime_files],
        "model": {
            "path": str(model),
            "bytes": model.stat().st_size,
            "sha256": config.model_sha256.lower(),
        },
        "config": {
            "n_predict": config.n_predict,
            "gpu_layers": config.gpu_layers,
            "threads": config.threads,
            "sampling": "greedy",
            "trace_backend": "vulkan",
        },
    }


def collect_trace_pairs(
    corpus_path: Path,
    config: CollectionConfig,
    *,
    process_runner: Any = _run_native_process,
    expected_split_counts: dict[str, int] | None = None,
    expected_domain_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Collect or resume parity-paired probe shards with exact provenance."""

    if config.n_predict <= 0 or config.gpu_layers < 0 or config.threads <= 0:
        raise ValueError("collection numeric settings are invalid")
    if not config.probe.is_file():
        raise FileNotFoundError(config.probe)
    if not config.model.is_file():
        raise FileNotFoundError(config.model)
    digest = config.model_sha256.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("model_sha256 must be 64 hexadecimal characters")
    if _sha256(config.model.resolve()) != digest:
        raise ValueError("model_sha256 does not match the model file")

    corpus = load_corpus_manifest(
        corpus_path,
        expected_split_counts=expected_split_counts,
        expected_domain_counts=expected_domain_counts,
    )
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "collection-manifest.json"
    identity = _identity(corpus_path, corpus, config)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("identity") != identity:
            raise ValueError("existing collection manifest identity does not match")
        report = existing
    else:
        report = {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "shards": [],
        }

    raw_shards = report.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("existing collection shards must be an array")
    shards_by_id = {
        shard.get("conversation_id"): shard
        for shard in raw_shards
        if isinstance(shard, dict)
        and isinstance(shard.get("conversation_id"), str)
    }
    passed = 0
    failed = 0
    skipped = 0
    for conversation in corpus.conversations:
        existing_shard = shards_by_id.get(conversation.conversation_id)
        if isinstance(existing_shard, dict) and _attempt_is_valid(existing_shard):
            passed += 1
            skipped += 1
            continue
        prior_attempts = (
            existing_shard.get("attempts", [])
            if isinstance(existing_shard, dict)
            else []
        )
        if not isinstance(prior_attempts, list):
            raise ValueError("existing shard attempts must be an array")
        attempt_index = len(prior_attempts)
        shard_root = output_dir / conversation.conversation_id
        shard_dir = (
            shard_root
            if attempt_index == 0
            else shard_root / f"attempt-{attempt_index:04d}"
        )
        shard_dir.mkdir(parents=True, exist_ok=True)
        baseline_tokens = shard_dir / "baseline-tokens.json"
        instrumented_tokens = shard_dir / "instrumented-tokens.json"
        trace_path = shard_dir / "trace.jsonl"
        baseline_stdout = shard_dir / "baseline.stdout.log"
        baseline_stderr = shard_dir / "baseline.stderr.log"
        instrumented_stdout = shard_dir / "instrumented.stdout.log"
        instrumented_stderr = shard_dir / "instrumented.stderr.log"
        parity_path = shard_dir / "parity.json"
        common = [
            str(config.probe.resolve()),
            "-m",
            str(config.model.resolve()),
            "-n",
            str(config.n_predict),
            "-ngl",
            str(config.gpu_layers),
            "--threads",
            str(config.threads),
        ]
        baseline_argv = common + [
            "--tokens",
            str(baseline_tokens),
            "--no-trace",
            conversation.prompt,
        ]
        instrumented_argv = common + [
            "--tokens",
            str(instrumented_tokens),
            "--trace",
            str(trace_path),
            conversation.prompt,
        ]
        baseline_result = process_runner(
            baseline_argv,
            cwd=config.probe.resolve().parent,
            stdout_path=baseline_stdout,
            stderr_path=baseline_stderr,
        )
        instrumented_result = process_runner(
            instrumented_argv,
            cwd=config.probe.resolve().parent,
            stdout_path=instrumented_stdout,
            stderr_path=instrumented_stderr,
        )
        attempt: dict[str, object] = {
            "attempt_index": attempt_index,
            "baseline": _process_record(
                baseline_argv,
                baseline_result,
                stdout_path=baseline_stdout,
                stderr_path=baseline_stderr,
                tokens_path=baseline_tokens,
            ),
            "instrumented": _process_record(
                instrumented_argv,
                instrumented_result,
                stdout_path=instrumented_stdout,
                stderr_path=instrumented_stderr,
                tokens_path=instrumented_tokens,
            ),
        }
        status = "process_failed"
        if (
            baseline_result.return_code == 0
            and instrumented_result.return_code == 0
            and trace_path.is_file()
            and baseline_tokens.is_file()
            and instrumented_tokens.is_file()
        ):
            parity_report = compare_token_sequences(
                baseline_tokens, instrumented_tokens
            )
            parity_path.write_text(
                json.dumps(parity_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            event_count = sum(1 for _ in load_router_events(trace_path))
            attempt["trace"] = {
                "artifact": _artifact(trace_path),
                "event_count": event_count,
            }
            attempt["parity"] = {
                **parity_report,
                "artifact": _artifact(parity_path),
            }
            status = (
                "passed"
                if event_count > 0
                and parity_report["prompt_matches"]
                and parity_report["generated_matches"]
                else "parity_failed"
            )
        shard = existing_shard if isinstance(existing_shard, dict) else {
            "conversation_id": conversation.conversation_id,
            "split": conversation.split,
            "domain": conversation.domain,
            "source_dataset": conversation.source_dataset,
            "source_record_id": conversation.source_record_id,
            "message_count": conversation.message_count,
            "attempts": [],
        }
        attempts = shard.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("existing shard attempts must be an array")
        attempts.append(attempt)
        shard["latest_status"] = status
        if conversation.conversation_id not in shards_by_id:
            raw_shards.append(shard)
            shards_by_id[conversation.conversation_id] = shard
        if status == "passed":
            passed += 1
        else:
            failed += 1
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        report["summary"] = {
            "conversation_count": len(corpus.conversations),
            "passed": passed,
            "failed": failed,
            "skipped_valid": skipped,
        }
        _write_json_atomic(manifest_path, report)

    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = {
        "conversation_count": len(corpus.conversations),
        "passed": passed,
        "failed": failed,
        "skipped_valid": skipped,
    }
    _write_json_atomic(manifest_path, report)
    return report
