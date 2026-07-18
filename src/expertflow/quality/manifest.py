from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SELECTION_SALT = "expertflow-option1-v1"
MMLU_SUBJECTS = (
    "abstract_algebra",
    "college_computer_science",
    "college_mathematics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "high_school_world_history",
    "machine_learning",
    "moral_scenarios",
    "professional_law",
)
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class FreezeConfig:
    output_path: Path
    wikitext_revision: str
    mmlu_revision: str
    wikitext_source: Path
    mmlu_source: Path
    model_path: Path
    llama_cli_path: Path
    llama_perplexity_path: Path
    wikitext_token_count: int
    mmlu_rows_by_subject: Mapping[str, Sequence[Mapping[str, Any]]]
    runtime: Mapping[str, Any]


def freeze_config_from_json(path: Path) -> FreezeConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    mmlu_source = Path(data["mmlu_source"])
    rows = json.loads(mmlu_source.read_text(encoding="utf-8"))
    return FreezeConfig(
        output_path=Path(data["output_path"]),
        wikitext_revision=str(data["wikitext_revision"]),
        mmlu_revision=str(data["mmlu_revision"]),
        wikitext_source=Path(data["wikitext_source"]),
        mmlu_source=mmlu_source,
        model_path=Path(data["model_path"]),
        llama_cli_path=Path(data["llama_cli_path"]),
        llama_perplexity_path=Path(data["llama_perplexity_path"]),
        wikitext_token_count=int(data["wikitext_token_count"]),
        mmlu_rows_by_subject=rows,
        runtime=dict(data["runtime"]),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_manifest_hash(manifest: Mapping[str, Any]) -> str:
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(unhashed)).hexdigest()


def _selection_key(subject: str, row_id: object) -> str:
    value = f"{SELECTION_SALT}:{subject}:{row_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_mmlu_items(
    rows_by_subject: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if set(rows_by_subject) != set(MMLU_SUBJECTS):
        missing = sorted(set(MMLU_SUBJECTS) - set(rows_by_subject))
        extra = sorted(set(rows_by_subject) - set(MMLU_SUBJECTS))
        raise ValueError(f"MMLU subjects mismatch: missing={missing}, extra={extra}")

    selected: list[dict[str, Any]] = []
    for subject in MMLU_SUBJECTS:
        rows = rows_by_subject[subject]
        ids = [row.get("row_id") for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate row_id in MMLU subject {subject}")
        if len(rows) < 10:
            raise ValueError(f"MMLU subject {subject} has fewer than 10 rows")
        ranked = sorted(rows, key=lambda row: _selection_key(subject, row["row_id"]))[:10]
        for row in ranked:
            choices = list(row["choices"])
            answer = int(row["answer"])
            if len(choices) != 4 or answer not in range(4):
                raise ValueError(f"invalid MMLU choices or answer for {subject}:{row['row_id']}")
            selected.append(
                {
                    "subject": subject,
                    "row_id": row["row_id"],
                    "selection_sha256": _selection_key(subject, row["row_id"]),
                    "question": str(row["question"]),
                    "choices": [str(choice) for choice in choices],
                    "answer": answer,
                }
            )
    return selected


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _write_once(path: Path, manifest: Mapping[str, Any]) -> None:
    rendered = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_manifest_hash(existing) != canonical_manifest_hash(manifest):
            raise FileExistsError(f"refusing to overwrite different frozen manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def freeze_manifest(config: FreezeConfig) -> dict[str, Any]:
    for name, revision in (
        ("wikitext", config.wikitext_revision),
        ("mmlu", config.mmlu_revision),
    ):
        if not _IMMUTABLE_REVISION.fullmatch(revision):
            raise ValueError(f"{name} revision must be an immutable 40-character lowercase SHA")
    if config.wikitext_token_count != 8192:
        raise ValueError("WikiText evaluation must contain exactly 8192 tokens")

    items = select_mmlu_items(config.mmlu_rows_by_subject)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "datasets": {
            "wikitext": {
                "repository": "Salesforce/wikitext",
                "configuration": "wikitext-2-raw-v1",
                "split": "test",
                "revision": config.wikitext_revision,
                "source": _file_identity(config.wikitext_source),
            },
            "mmlu": {
                "repository": "cais/mmlu",
                "split": "test",
                "revision": config.mmlu_revision,
                "source": _file_identity(config.mmlu_source),
            },
        },
        "executables": {
            "llama_cli": _file_identity(config.llama_cli_path),
            "llama_perplexity": _file_identity(config.llama_perplexity_path),
        },
        "model": _file_identity(config.model_path),
        "runtime": dict(config.runtime),
        "thresholds": {
            "perplexity_relative_max": 0.005,
            "mmlu_accuracy_delta_pp_min": -1.0,
            "repeated_4gram_delta_pp_max": 5.0,
            "distinct_2_delta_pp_min": -5.0,
        },
        "wikitext": {"token_count": 8192, "chunk_tokens": 2048, "chunk_count": 4},
        "mmlu": {
            "selection_salt": SELECTION_SALT,
            "item_count": len(items),
            "items": items,
        },
    }
    manifest["manifest_sha256"] = canonical_manifest_hash(manifest)
    _write_once(config.output_path, manifest)
    return manifest
