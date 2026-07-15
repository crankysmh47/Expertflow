from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertflow.expanded_collection import (
    ExpandedManifestError,
    load_expanded_manifest,
    normalized_prompt_sha256,
    prompt_sha256,
    select_collection_rows,
    validate_canonical_shard,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "expanded-canonical-84.json"


def test_frozen_manifest_has_exact_split_and_per_domain_allocation() -> None:
    manifest = load_expanded_manifest(MANIFEST)

    assert manifest.dataset_id == "trace-v2-canonical-expanded-84-v1"
    assert len(manifest.conversations) == 84
    assert manifest.split_counts == {"train": 60, "validation": 12, "test": 12}
    for domain in manifest.domains:
        rows = [row for row in manifest.conversations if row.domain == domain]
        assert len([row for row in rows if row.split == "train"]) == 10
        assert len([row for row in rows if row.split == "validation"]) == 2
        assert len([row for row in rows if row.split == "test"]) == 2


def test_frozen_manifest_hashes_and_deduplication_contract_are_valid() -> None:
    manifest = load_expanded_manifest(MANIFEST)

    assert len({row.conversation_id for row in manifest.conversations}) == 84
    assert len({row.template_id for row in manifest.conversations}) == 84
    assert len({row.prompt_sha256 for row in manifest.conversations}) == 84
    assert len({row.normalized_prompt_sha256 for row in manifest.conversations}) == 84
    assert all(row.prompt_sha256 == prompt_sha256(row.prompt) for row in manifest.conversations)
    assert all(
        row.normalized_prompt_sha256 == normalized_prompt_sha256(row.prompt)
        for row in manifest.conversations
    )


def test_task_families_never_cross_splits() -> None:
    manifest = load_expanded_manifest(MANIFEST)
    family_splits: dict[str, set[str]] = {}
    for row in manifest.conversations:
        family_splits.setdefault(row.task_family, set()).add(row.split)

    assert all(len(splits) == 1 for splits in family_splits.values())


def test_loader_rejects_a_cross_split_superficial_variant(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = next(row for row in payload["conversations"] if row["split"] == "train")
    validation = next(
        row for row in payload["conversations"] if row["split"] == "validation"
    )
    validation["task_family"] = train["task_family"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExpandedManifestError, match="task_family crosses splits"):
        load_expanded_manifest(invalid)


def test_loader_rejects_a_stale_prompt_hash(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["conversations"][0]["prompt"] += " changed"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExpandedManifestError, match="prompt_sha256"):
        load_expanded_manifest(invalid)


def _event(*, layer: int, hook: int, conversation_id: str = "exp84-tr-code-01") -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "request_id": f"request-{conversation_id}",
        "conversation_id": conversation_id,
        "turn_index": 0,
        "phase": "decode",
        "forward_id": 7,
        "hook_order": hook,
        "token_index": 12,
        "token_id": 42,
        "layer_id": layer,
        "selected_expert_ids": list(range(8)),
        "selected_expert_weights": None,
        "observed_at_ns": 1000 + hook,
    }


def test_shard_validator_requires_complete_ordered_30_layer_forwards(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(_event(layer=layer, hook=layer)) + "\n" for layer in range(30)),
        encoding="utf-8",
    )

    summary = validate_canonical_shard(trace, "exp84-tr-code-01")

    assert summary == {"event_count": 30, "forward_count": 1, "layer_count": 30}


def test_shard_validator_rejects_incomplete_or_wrong_conversation(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    events = [_event(layer=layer, hook=layer) for layer in range(29)]
    events[0]["conversation_id"] = "wrong"
    trace.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    with pytest.raises(ExpandedManifestError):
        validate_canonical_shard(trace, "exp84-tr-code-01")


def test_collection_selection_keeps_test_sealed_and_round_robins_domains() -> None:
    manifest = load_expanded_manifest(MANIFEST)

    selected = select_collection_rows(manifest, splits=("train", "validation"))

    assert len(selected) == 72
    assert not any(row.split == "test" for row in selected)
    assert [row.domain for row in selected[:6]] == list(manifest.domains)


def test_collection_selection_requires_explicit_test_unseal() -> None:
    manifest = load_expanded_manifest(MANIFEST)

    with pytest.raises(ExpandedManifestError, match="sealed"):
        select_collection_rows(manifest, splits=("test",))
