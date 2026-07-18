from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertflow.quality.manifest import (
    FreezeConfig,
    canonical_manifest_hash,
    freeze_config_from_json,
    freeze_manifest,
    select_mmlu_items,
)


SUBJECTS = (
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


def _write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8")
    return path


def _rows_by_subject() -> dict[str, list[dict[str, object]]]:
    return {
        subject: [
            {
                "row_id": index,
                "question": f"{subject} question {index}",
                "choices": ["A0", "B0", "C0", "D0"],
                "answer": index % 4,
            }
            for index in range(15)
        ]
        for subject in SUBJECTS
    }


def _fixture_config(tmp_path: Path) -> FreezeConfig:
    return FreezeConfig(
        output_path=tmp_path / "manifest.json",
        wikitext_revision="a" * 40,
        mmlu_revision="b" * 40,
        wikitext_source=_write(tmp_path / "wikitext.txt", "one\ntwo\n"),
        mmlu_source=_write(tmp_path / "mmlu.json", json.dumps(_rows_by_subject())),
        model_path=_write(tmp_path / "model.gguf", "model"),
        llama_cli_path=_write(tmp_path / "llama-cli.exe", "cli"),
        llama_perplexity_path=_write(tmp_path / "llama-perplexity.exe", "ppl"),
        wikitext_token_count=8192,
        mmlu_rows_by_subject=_rows_by_subject(),
        runtime={"seed": 1, "temperature": 0, "ngl": 10},
    )


def test_freeze_manifest_is_stable_and_content_addressed(tmp_path: Path):
    manifest = freeze_manifest(_fixture_config(tmp_path))

    assert manifest["schema_version"] == 1
    assert len(manifest["mmlu"]["items"]) == 100
    assert manifest["mmlu"]["selection_salt"] == "expertflow-option1-v1"
    assert manifest["wikitext"]["token_count"] == 8192
    assert manifest["manifest_sha256"] == canonical_manifest_hash(manifest)


def test_select_mmlu_items_is_order_independent():
    rows = _rows_by_subject()
    forward = select_mmlu_items(rows)
    reverse = select_mmlu_items(
        {subject: list(reversed(subject_rows)) for subject, subject_rows in reversed(rows.items())}
    )

    assert forward == reverse
    assert len(forward) == 100


def test_freeze_manifest_refuses_mutable_revisions(tmp_path: Path):
    config = _fixture_config(tmp_path)
    config = FreezeConfig(**{**config.__dict__, "wikitext_revision": "main"})

    with pytest.raises(ValueError, match="immutable 40-character"):
        freeze_manifest(config)


def test_freeze_manifest_refuses_changed_overwrite(tmp_path: Path):
    config = _fixture_config(tmp_path)
    freeze_manifest(config)
    config.mmlu_rows_by_subject[SUBJECTS[0]][0]["question"] = "changed"

    with pytest.raises(FileExistsError, match="different frozen manifest"):
        freeze_manifest(config)


def test_freeze_config_from_json_loads_mmlu_rows(tmp_path: Path):
    config = _fixture_config(tmp_path)
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
        if key != "mmlu_rows_by_subject"
    }
    config_path = tmp_path / "freeze-config.json"
    config_path.write_text(json.dumps(serialized), encoding="utf-8")

    loaded = freeze_config_from_json(config_path)

    assert loaded.mmlu_rows_by_subject == _rows_by_subject()
    assert loaded.output_path == config.output_path
