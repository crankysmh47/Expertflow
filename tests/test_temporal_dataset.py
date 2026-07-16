from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertflow.predictor.temporal_dataset import load_temporal_dataset


def _event(
    conversation: str,
    forward: int,
    *,
    phase: str,
    layer: int = 24,
    token_index: int | None = None,
    request_id: str | None = None,
    turn_index: int = 0,
    hook_order: int | None = None,
    experts: tuple[int, ...] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "request_id": request_id or f"request-{conversation}",
        "conversation_id": conversation,
        "turn_index": turn_index,
        "phase": phase,
        "forward_id": forward,
        "hook_order": hook_order if hook_order is not None else forward * 30 + layer,
        "token_index": forward if token_index is None else token_index,
        "token_id": 1000 + forward,
        "layer_id": layer,
        "selected_expert_ids": list(experts or tuple(range(forward, forward + 8))),
        "selected_expert_weights": None,
        "observed_at_ns": 10_000 + forward * 30 + layer,
    }


def _manifest(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    shards = []
    for conversation, split in rows:
        trace = tmp_path / f"{conversation}.jsonl"
        events = [
            _event(conversation, 0, phase="prefill"),
            _event(conversation, 10, phase="decode", token_index=10),
            _event(conversation, 11, phase="decode", token_index=11),
            _event(conversation, 12, phase="decode", token_index=12),
        ]
        trace.write_text(
            "".join(json.dumps(event) + "\n" for event in reversed(events)),
            encoding="utf-8",
        )
        shards.append({
            "conversation_id": conversation,
            "split": split,
            "domain": "domain-a",
            "prompt_sha256": f"prompt-{conversation}",
            "status": "passed",
            "trace": {"path": str(trace)},
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "canonical_runtime": "expertflow-canonical-observer-v1",
        "trace_generation": "trace_v2_canonical_segmented",
        "shards": shards,
    }), encoding="utf-8")
    return path


def test_builds_decode_only_consecutive_layer24_pairs(tmp_path: Path) -> None:
    dataset = load_temporal_dataset(
        _manifest(tmp_path, [
            ("train-a", "train"),
            ("validation-a", "validation"),
            ("test-a", "test"),
        ]),
        expected_split_counts={"train": 1, "validation": 1, "test": 1},
    )

    assert len(dataset.train) == 2
    first, second = dataset.train
    assert first.phase == "decode"
    assert first.layer_id == 24
    assert (first.source_forward_id, first.target_forward_id) == (10, 11)
    assert (first.source_token_index, first.target_token_index) == (10, 11)
    assert first.source_expert_ids == tuple(range(10, 18))
    assert first.target_expert_ids == tuple(range(11, 19))
    assert second.source_expert_ids == first.target_expert_ids
    assert dataset.conversation_ids["test"] == ("test-a",)


def test_validates_sealed_identity_without_reading_test_trace(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("train-a", "train"), ("test-a", "test")])
    payload = json.loads(manifest.read_text())
    Path(payload["shards"][1]["trace"]["path"]).unlink()

    dataset = load_temporal_dataset(
        manifest,
        expected_split_counts={"train": 1, "test": 1},
        materialize_splits={"train"},
    )

    assert len(dataset.train) == 2
    assert dataset.test == ()
    assert dataset.conversation_ids["test"] == ("test-a",)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "duplicate layer-24"),
        ("forward_gap", "consecutive forward"),
        ("token_gap", "consecutive token"),
        ("request", "request changed"),
        ("turn", "turn changed"),
        ("hook", "causal hook"),
        ("width", "exactly eight"),
        ("range", "outside"),
    ],
)
def test_rejects_ambiguous_temporal_sequences(
    tmp_path: Path, mutation: str, match: str
) -> None:
    manifest = _manifest(tmp_path, [("train-a", "train")])
    payload = json.loads(manifest.read_text())
    trace = Path(payload["shards"][0]["trace"]["path"])
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    decode = sorted((e for e in events if e["phase"] == "decode"), key=lambda e: e["forward_id"])
    if mutation == "duplicate":
        events.append(dict(decode[0]))
    elif mutation == "forward_gap":
        decode[1]["forward_id"] = 13
    elif mutation == "token_gap":
        decode[1]["token_index"] = 13
    elif mutation == "request":
        decode[1]["request_id"] = "other"
    elif mutation == "turn":
        decode[1]["turn_index"] = 1
    elif mutation == "hook":
        decode[1]["hook_order"] = decode[0]["hook_order"]
    elif mutation == "width":
        decode[1]["selected_expert_ids"] = [1, 2]
    else:
        decode[1]["selected_expert_ids"][0] = 128
    trace.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_temporal_dataset(manifest, expected_split_counts={"train": 1})


def test_enforces_frozen_split_domain_and_prompt_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [
        ("train-a", "train"),
        ("validation-a", "validation"),
        ("test-a", "test"),
    ])
    dataset = load_temporal_dataset(
        manifest,
        expected_split_counts={"train": 1, "validation": 1, "test": 1},
        expected_domain_counts={
            "train": {"domain-a": 1},
            "validation": {"domain-a": 1},
            "test": {"domain-a": 1},
        },
        require_unique_prompt_hashes=True,
    )
    assert len(dataset.validation) == 2

    payload = json.loads(manifest.read_text())
    payload["shards"][1]["prompt_sha256"] = payload["shards"][0]["prompt_sha256"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt hash"):
        load_temporal_dataset(
            manifest,
            expected_split_counts={"train": 1, "validation": 1, "test": 1},
            require_unique_prompt_hashes=True,
        )

