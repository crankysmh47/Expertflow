import json
from pathlib import Path

import pytest

from expertflow.analysis.heldout_breakdown import (
    EvaluationTrace,
    build_heldout_breakdown,
    load_collection_breakdown_inputs,
)
from expertflow.trace.schema import RouterTraceEvent


def event(
    forward_id: int, layer_id: int, experts: tuple[int, ...]
) -> RouterTraceEvent:
    return RouterTraceEvent(
        schema_version="1.0.0",
        request_id="req-1",
        conversation_id="conv-1",
        turn_index=0,
        phase="decode",
        forward_id=forward_id,
        hook_order=forward_id * 2 + layer_id,
        token_index=forward_id,
        token_id=100 + forward_id,
        layer_id=layer_id,
        selected_expert_ids=experts,
        selected_expert_weights=None,
        observed_at_ns=(forward_id * 2 + layer_id) * 1_000,
    )


def test_freezes_static_training_and_resets_lru_per_conversation() -> None:
    training = [
        event(forward, layer, (1, 2))
        for forward in range(2)
        for layer in (0, 1)
    ]
    evaluations = [
        EvaluationTrace(
            conversation_id="validation-code",
            split="validation",
            domain="code",
            source_trace="validation.jsonl",
            events=(event(10, 0, (3, 4)), event(10, 1, (3, 4))),
        ),
        EvaluationTrace(
            conversation_id="test-code",
            split="test",
            domain="code",
            source_trace="test.jsonl",
            events=(event(20, 0, (3, 4)), event(20, 1, (3, 4))),
        ),
    ]

    report = build_heldout_breakdown(
        training,
        evaluations,
        capacity_per_layer=2,
        slot_bytes=100,
        expert_transfer_ms=0.5,
    )

    assert report["fit_scope"] == "held_out_conversation_split"
    assert report["lru_reset_scope"] == "conversation"
    assert report["training_event_count"] == 4
    assert report["evaluation_event_count"] == 4
    assert len(report["per_prompt"]) == 2
    for prompt in report["per_prompt"]:
        assert prompt["token_count"] == 1
        assert prompt["static_hotset"]["hit_rate"] == 0.0
        assert prompt["lru"]["hit_rate"] == 0.0
        assert prompt["static_hotset"]["cold_bytes_per_token"] == 400
        assert prompt["static_hotset"][
            "serialized_transfer_ms_per_token"
        ] == pytest.approx(2.0)
    domain = report["per_domain"][0]
    assert domain["domain"] == "code"
    assert domain["conversation_count"] == 2
    assert domain["static_hotset"]["demand_count"] == 8
    assert domain["lru"]["hit_count"] == 0
    assert report["aggregate"]["static_hotset"] == domain["static_hotset"]


def test_rejects_evaluation_layer_mismatch() -> None:
    with pytest.raises(ValueError, match="layer sets"):
        build_heldout_breakdown(
            [event(0, 0, (1, 2)), event(0, 1, (1, 2))],
            [
                EvaluationTrace(
                    conversation_id="bad",
                    split="test",
                    domain="code",
                    source_trace="bad.jsonl",
                    events=(event(1, 0, (1, 2)),),
                )
            ],
            capacity_per_layer=2,
            slot_bytes=100,
            expert_transfer_ms=0.5,
        )


def test_loads_passed_collection_shards_by_complete_split(
    tmp_path: Path,
) -> None:
    trace_record = {
        "schema_version": "1.0.0",
        "request_id": "req-001",
        "conversation_id": "conv-001",
        "turn_index": 0,
        "phase": "decode",
        "forward_id": 1,
        "hook_order": 0,
        "token_index": 1,
        "token_id": 100,
        "layer_id": 0,
        "selected_expert_ids": [1, 2],
        "selected_expert_weights": None,
        "observed_at_ns": 1,
    }
    train_trace = tmp_path / "train.jsonl"
    test_trace = tmp_path / "test.jsonl"
    for path in (train_trace, test_trace):
        path.write_text(json.dumps(trace_record) + "\n", encoding="utf-8")
    manifest = tmp_path / "collection-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "conversation_id": "train-1",
                        "split": "train",
                        "domain": "code",
                        "latest_status": "passed",
                        "attempts": [
                            {
                                "trace": {
                                    "artifact": {"path": str(train_trace)}
                                }
                            }
                        ],
                    },
                    {
                        "conversation_id": "test-1",
                        "split": "test",
                        "domain": "math_reasoning",
                        "latest_status": "passed",
                        "attempts": [
                            {
                                "trace": {
                                    "artifact": {"path": str(test_trace)}
                                }
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    training, evaluations, excluded = load_collection_breakdown_inputs(
        manifest, phase="decode", max_layer=0
    )

    assert len(training) == 1
    assert len(evaluations) == 1
    assert evaluations[0].conversation_id == "test-1"
    assert evaluations[0].split == "test"
    assert evaluations[0].domain == "math_reasoning"
    assert evaluations[0].source_trace == str(test_trace.resolve())
    assert excluded == ()


def test_failed_shard_requires_explicit_audited_exclusion(
    tmp_path: Path,
) -> None:
    trace_record = {
        "schema_version": "1.0.0",
        "request_id": "req-001",
        "conversation_id": "conv-001",
        "turn_index": 0,
        "phase": "decode",
        "forward_id": 1,
        "hook_order": 0,
        "token_index": 1,
        "token_id": 100,
        "layer_id": 0,
        "selected_expert_ids": [1, 2],
        "selected_expert_weights": None,
        "observed_at_ns": 1,
    }
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(trace_record) + "\n", encoding="utf-8")
    manifest = tmp_path / "collection-manifest.json"
    passed_attempt = {"trace": {"artifact": {"path": str(trace)}}}
    manifest.write_text(
        json.dumps(
            {
                "summary": {
                    "conversation_count": 3,
                    "passed": 2,
                    "failed": 1,
                },
                "shards": [
                    {
                        "conversation_id": "train-ok",
                        "split": "train",
                        "domain": "code",
                        "latest_status": "passed",
                        "attempts": [passed_attempt],
                    },
                    {
                        "conversation_id": "train-failed",
                        "split": "train",
                        "domain": "translation",
                        "latest_status": "parity_failed",
                        "attempts": [{"parity": {"generated_matches": False}}],
                    },
                    {
                        "conversation_id": "test-ok",
                        "split": "test",
                        "domain": "math_reasoning",
                        "latest_status": "passed",
                        "attempts": [passed_attempt],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="failed shards"):
        load_collection_breakdown_inputs(
            manifest, phase="decode", max_layer=0
        )

    training, evaluations, excluded = load_collection_breakdown_inputs(
        manifest,
        phase="decode",
        max_layer=0,
        exclude_failed=True,
    )

    assert len(training) == 1
    assert len(evaluations) == 1
    assert excluded == (
        {
            "conversation_id": "train-failed",
            "split": "train",
            "domain": "translation",
            "latest_status": "parity_failed",
            "attempt_count": 1,
        },
    )
