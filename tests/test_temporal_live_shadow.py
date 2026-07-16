from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertflow.predictor.temporal_live_shadow import (
    load_temporal_shadow_log,
    summarize_temporal_shadow,
    validate_temporal_offline_equivalence,
)
from expertflow.predictor.temporal_runtime_artifact import (
    TemporalArtifactIdentity,
    build_temporal_runtime_artifact,
    parse_temporal_runtime_artifact,
    predict_temporal_runtime_artifact,
)
from expertflow.predictor.temporal_models import TemporalTransitionPredictor
from expertflow.predictor.temporal_dataset import TemporalSample


def _sample(source, target):
    return TemporalSample(
        "c", "train", "d", "r", 0, "decode", 24,
        1, 2, 1, 2, 1, 2, 1, 2, source, target,
    )


def _artifact():
    predictor = TemporalTransitionPredictor.fit((
        _sample(tuple(range(8)), tuple(range(20, 28))),
        _sample(tuple(range(20, 28)), tuple(range(40, 48))),
    ))
    identity = TemporalArtifactIdentity(*("11" * 32, "22" * 32, "33" * 32, "44" * 32))
    return parse_temporal_runtime_artifact(
        build_temporal_runtime_artifact(predictor, identity)
    )


def test_log_validates_offline_scores_and_summarizes_deadlines(tmp_path: Path) -> None:
    artifact = _artifact()
    source = tuple(range(8))
    predicted, scores, counts = predict_temporal_runtime_artifact(
        artifact, source_expert_ids=source, session_counts={}
    )
    actual = tuple(range(20, 28))
    record = {
        "schema_version": "1.0.0",
        "record_kind": "transition",
        "run_id": "run",
        "conversation_generation": 1,
        "source_forward_index": 10,
        "target_forward_index": 11,
        "source_decode_index": 0,
        "target_decode_index": 1,
        "source_observed_ns": 1_000_000,
        "predictor_finished_ns": 1_080_000,
        "target_observed_ns": 3_000_000,
        "prediction_latency_ns": 80_000,
        "source_experts": source,
        "session_counts_after_source": [counts[index] for index in range(128)],
        "predicted_experts": predicted,
        "predicted_scores": scores,
        "actual_experts": actual,
        "artifact_sha256": artifact.payload_sha256,
        "configuration_sha256": artifact.identity.configuration_sha256,
    }
    path = tmp_path / "shadow.jsonl"
    path.write_text(
        json.dumps(record) + "\n" +
        json.dumps({
            "schema_version": "1.0.0",
            "record_kind": "summary",
            "run_id": "run",
            "transitions": 1,
            "pending_prediction": True,
            "conversation_generation": 1,
        }) + "\n",
        encoding="utf-8",
    )
    records, summary = load_temporal_shadow_log(path)
    validate_temporal_offline_equivalence(artifact, records)
    measured = summarize_temporal_shadow(records, summary)
    assert measured["transitions"] == 1
    assert measured["prediction_latency_us"]["p50"] == 80.0
    assert measured["lead_time_us"]["p50"] == 1920.0
    assert measured["deadline_eligible"]["h2d_only"] == 1
    assert measured["ranking"]["hit_at_1"] in {0, 1}


def test_log_rejects_skipped_decode_identity(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    rows = [
        {
            "schema_version": "1.0.0", "record_kind": "transition",
            "run_id": "run", "conversation_generation": 1,
            "source_forward_index": 1, "target_forward_index": 3,
            "source_decode_index": 0, "target_decode_index": 2,
            "source_observed_ns": 1, "predictor_finished_ns": 2,
            "target_observed_ns": 3, "prediction_latency_ns": 1,
            "source_experts": list(range(8)),
            "session_counts_after_source": [0] * 128,
            "predicted_experts": list(range(16)),
            "predicted_scores": [1.0] * 16,
            "actual_experts": list(range(8)),
            "artifact_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
        },
        {
            "schema_version": "1.0.0", "record_kind": "summary",
            "run_id": "run", "transitions": 1,
            "pending_prediction": True, "conversation_generation": 1,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="consecutive"):
        load_temporal_shadow_log(path)

