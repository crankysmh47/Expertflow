from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertflow.predictor.dataset import load_pilot_dataset


def _event(conversation: str, token: int, layer: int, *, expert: int | None = None) -> dict[str, object]:
    selected = expert if expert is not None else token * 10 + layer
    return {
        "schema_version": "1.0.0",
        "request_id": f"request-{conversation}",
        "conversation_id": conversation,
        "turn_index": 0,
        "phase": "prefill" if token == 0 else "decode",
        "forward_id": token,
        "hook_order": token * 3 + layer,
        "token_index": token,
        "token_id": 100 + token,
        "layer_id": layer,
        "selected_expert_ids": [selected, selected + 32],
        "selected_expert_weights": None,
        "observed_at_ns": 1000 + token * 3 + layer,
    }


def _manifest(tmp_path: Path, rows: list[tuple[str, str]], *, domains: dict[str, str] | None = None) -> Path:
    shards = []
    for conversation, split in rows:
        trace = tmp_path / f"{conversation}.jsonl"
        events = [_event(conversation, token, layer) for token in range(2) for layer in range(3)]
        trace.write_text("".join(json.dumps(event) + "\n" for event in reversed(events)), encoding="utf-8")
        shards.append({
            "conversation_id": conversation,
            "split": split,
            "domain": (domains or {}).get(conversation, "test-domain"),
            "prompt_sha256": f"prompt-{conversation}",
            "status": "passed",
            "trace": {"path": str(trace)},
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "1.0.0",
        "canonical_runtime": "expertflow-canonical-observer-v1",
        "trace_generation": "trace_v2_canonical_segmented",
        "shards": shards,
    }), encoding="utf-8")
    return manifest


def test_builds_adjacent_samples_and_causal_previous_target(tmp_path: Path) -> None:
    dataset = load_pilot_dataset(_manifest(tmp_path, [
        ("train-a", "train"),
        ("validation-a", "validation"),
        ("test-a", "test"),
    ]), expected_split_counts={"train": 1, "validation": 1, "test": 1})

    assert len(dataset.train) == 4
    first, second_transition, next_token = dataset.train[0], dataset.train[1], dataset.train[2]
    assert (first.source_layer, first.target_layer) == (0, 1)
    assert first.source_expert_ids == (0, 32)
    assert first.target_expert_ids == (1, 33)
    assert first.previous_target_vector is None
    assert second_transition.target_layer == 2
    assert next_token.previous_target_vector is not None
    assert next_token.previous_target_vector[1] == 1.0
    assert sum(next_token.source_vector) == 2.0
    assert dataset.conversation_ids == {
        "train": ("train-a",),
        "validation": ("validation-a",),
        "test": ("test-a",),
    }


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate layer event"),
    ("missing", "incomplete layer sequence"),
    ("conversation", "conversation mismatch"),
])
def test_rejects_ambiguous_or_incomplete_joins(tmp_path: Path, mutation: str, match: str) -> None:
    manifest = _manifest(tmp_path, [("train-a", "train")])
    payload = json.loads(manifest.read_text())
    trace = Path(payload["shards"][0]["trace"]["path"])
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    if mutation == "duplicate":
        events.append(events[0])
    elif mutation == "missing":
        events.pop()
    else:
        events[0]["conversation_id"] = "other"
    trace.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_pilot_dataset(manifest, expected_split_counts={"train": 1})


def test_rejects_conversation_in_more_than_one_split(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("same", "train")])
    payload = json.loads(manifest.read_text())
    duplicate = dict(payload["shards"][0])
    duplicate["split"] = "test"
    payload["shards"].append(duplicate)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="multiple splits"):
        load_pilot_dataset(manifest, expected_split_counts={"train": 1, "test": 1})


def test_enforces_domain_balance_and_unique_prompt_hashes(tmp_path: Path) -> None:
    rows = [("train-a", "train"), ("validation-a", "validation"), ("test-a", "test")]
    manifest = _manifest(tmp_path, rows, domains={conversation: "domain-a" for conversation, _ in rows})

    dataset = load_pilot_dataset(
        manifest,
        expected_split_counts={"train": 1, "validation": 1, "test": 1},
        expected_domain_counts={
            "train": {"domain-a": 1},
            "validation": {"domain-a": 1},
            "test": {"domain-a": 1},
        },
        require_unique_prompt_hashes=True,
    )
    assert len(dataset.train) == 4

    payload = json.loads(manifest.read_text())
    payload["shards"][1]["prompt_sha256"] = payload["shards"][0]["prompt_sha256"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt hash"):
        load_pilot_dataset(
            manifest,
            expected_split_counts={"train": 1, "validation": 1, "test": 1},
            require_unique_prompt_hashes=True,
        )


def test_rejects_domain_count_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("train-a", "train")], domains={"train-a": "domain-a"})
    with pytest.raises(ValueError, match="domain counts"):
        load_pilot_dataset(
            manifest,
            expected_split_counts={"train": 1},
            expected_domain_counts={"train": {"domain-b": 1}},
        )


def test_can_validate_sealed_split_identity_without_opening_its_trace(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("train-a", "train"), ("test-a", "test")])
    payload = json.loads(manifest.read_text())
    Path(payload["shards"][1]["trace"]["path"]).unlink()

    dataset = load_pilot_dataset(
        manifest,
        expected_split_counts={"train": 1, "test": 1},
        materialize_splits={"train"},
    )

    assert len(dataset.train) == 4
    assert dataset.test == ()
    assert dataset.conversation_ids["test"] == ("test-a",)
