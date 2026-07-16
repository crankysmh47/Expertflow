from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertflow.predictor.dataset import PredictionSample
from expertflow.predictor.live_shadow import (
    load_shadow_log,
    summarize_shadow_records,
    validate_offline_equivalence,
    validate_token_and_router_parity,
)
from expertflow.predictor.models import TransitionPredictor
from expertflow.predictor.runtime_artifact import (
    ArtifactIdentity,
    build_runtime_artifact,
    parse_runtime_artifact,
    predict_runtime_artifact,
)


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _artifact():
    samples = []
    for phase, base in (("prefill", 20), ("decode", 40)):
        samples.append(
            PredictionSample(
                conversation_id=f"train-{phase}",
                split="train",
                domain="general_instruction",
                phase=phase,
                forward_id=0,
                token_index=0,
                token_id=1,
                source_layer=23,
                target_layer=24,
                source_expert_ids=(0, 1, 2, 3, 4, 5, 6, 7),
                target_expert_ids=tuple(range(base, base + 8)),
                source_expert_weights=None,
                source_vector=(0.0,) * 128,
                previous_target_vector=None,
            )
        )
        samples.append(
            PredictionSample(
                conversation_id=f"train-{phase}-extra",
                split="train",
                domain="general_instruction",
                phase=phase,
                forward_id=1,
                token_index=1,
                token_id=1,
                source_layer=23,
                target_layer=24,
                source_expert_ids=(0, 1, 8, 9, 10, 11, 12, 13),
                target_expert_ids=(
                    base,
                    base + 1,
                    base + 8,
                    base + 9,
                    base + 10,
                    base + 11,
                    base + 12,
                    base + 13,
                ),
                source_expert_weights=None,
                source_vector=(0.0,) * 128,
                previous_target_vector=None,
            )
        )
    predictor = TransitionPredictor.fit(
        samples, weighting="source_normalized", phase_mode="separate"
    )
    identity = ArtifactIdentity(
        model_sha256="11" * 32,
        runtime_sha256="22" * 32,
        manifest_sha256="33" * 32,
        configuration_sha256="44" * 32,
    )
    return parse_runtime_artifact(build_runtime_artifact(predictor, identity))


def test_live_shadow_log_matches_offline_artifact(tmp_path: Path) -> None:
    artifact = _artifact()
    source = (0, 1, 2, 3, 4, 5, 6, 7)
    predicted, scores = predict_runtime_artifact(
        artifact, phase="prefill", source_expert_ids=source
    )
    actual = tuple(range(20, 28))
    transition = {
        "schema_version": "1.0.0",
        "record_kind": "transition",
        "run_id": "run-1",
        "forward_index": 0,
        "phase": "prefill",
        "phase_generation": 1,
        "source_layer": 23,
        "target_layer": 24,
        "source_experts": source,
        "predicted_experts": predicted,
        "predicted_scores": scores,
        "actual_experts": actual,
        "recall_at_8_matches": 8,
        "recall_at_12_matches": 8,
        "recall_at_8": 1.0,
        "recall_at_12": 1.0,
        "prediction_latency_ns": 9000,
        "artifact_sha256": artifact.payload_sha256,
        "configuration_sha256": artifact.identity.configuration_sha256,
    }
    log = _write(
        tmp_path / "shadow.jsonl",
        [
            json.dumps(transition, separators=(",", ":")),
            (
                '{"schema_version":"1.0.0","record_kind":"summary",'
                '"run_id":"run-1","transitions":1,'
                '"candidate_support_failures":0,"pending_transition":false}'
            ),
        ],
    )
    records, summary = load_shadow_log(log)
    validate_offline_equivalence(artifact, records)
    measured = summarize_shadow_records(records, summary)
    assert measured["transitions"] == 1
    assert measured["prefill"]["latency_p50_us"] == 9.0
    assert measured["prefill"]["recall_at_12"] == 1.0


def test_shadow_log_rejects_incomplete_or_duplicate_transitions(
    tmp_path: Path,
) -> None:
    incomplete = _write(
        tmp_path / "incomplete.jsonl",
        [
            (
                '{"schema_version":"1.0.0","record_kind":"summary",'
                '"run_id":"run-1","transitions":0,'
                '"candidate_support_failures":0,"pending_transition":true}'
            )
        ],
    )
    with pytest.raises(ValueError, match="pending"):
        load_shadow_log(incomplete)


def test_token_and_router_parity_is_exact(tmp_path: Path) -> None:
    disabled_tokens = tmp_path / "disabled-tokens.json"
    enabled_tokens = tmp_path / "enabled-tokens.json"
    token_payload = (
        '{"schema_version":"1.0.0","prompt_token_ids":[1,2],'
        '"generated_token_ids":[3],"generated_text":"x"}\n'
    )
    disabled_tokens.write_text(token_payload, encoding="utf-8")
    enabled_tokens.write_text(token_payload, encoding="utf-8")
    trace_payload = (
        '{"phase":"prefill","forward_id":0,"token_index":0,"token_id":1,'
        '"layer_id":23,"selected_expert_ids":[1,2,3,4,5,6,7,8]}\n'
    )
    disabled_trace = tmp_path / "disabled-trace.jsonl"
    enabled_trace = tmp_path / "enabled-trace.jsonl"
    disabled_trace.write_text(trace_payload, encoding="utf-8")
    enabled_trace.write_text(trace_payload, encoding="utf-8")

    validate_token_and_router_parity(
        disabled_tokens,
        enabled_tokens,
        disabled_trace,
        enabled_trace,
    )

    enabled_trace.write_text(
        trace_payload.replace("[1,2,3,4,5,6,7,8]", "[8,2,3,4,5,6,7,1]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="router"):
        validate_token_and_router_parity(
            disabled_tokens,
            enabled_tokens,
            disabled_trace,
            enabled_trace,
        )
