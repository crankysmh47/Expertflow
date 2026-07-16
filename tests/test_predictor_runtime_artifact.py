from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from expertflow.predictor.dataset import PredictionSample
from expertflow.predictor.models import TransitionPredictor
from expertflow.predictor.runtime_artifact import (
    ArtifactIdentity,
    artifact_identity_payload,
    build_runtime_artifact,
    parse_runtime_artifact,
    predict_runtime_artifact,
    write_parity_fixtures,
)


def _sample(
    *,
    phase: str,
    source: tuple[int, ...],
    target: tuple[int, ...],
) -> PredictionSample:
    return PredictionSample(
        conversation_id=f"train-{phase}",
        split="train",
        domain="general_instruction",
        phase=phase,
        forward_id=1,
        token_index=1,
        token_id=1,
        source_layer=23,
        target_layer=24,
        source_expert_ids=source,
        target_expert_ids=target,
        source_expert_weights=None,
        source_vector=(0.0,) * 128,
        previous_target_vector=None,
    )


def _predictor() -> TransitionPredictor:
    samples = (
        _sample(
            phase="prefill",
            source=(0, 1, 2, 3, 4, 5, 6, 7),
            target=(20, 21, 22, 23, 24, 25, 26, 27),
        ),
        _sample(
            phase="prefill",
            source=(0, 1, 8, 9, 10, 11, 12, 13),
            target=(20, 21, 30, 31, 32, 33, 34, 35),
        ),
        _sample(
            phase="decode",
            source=(0, 1, 2, 3, 4, 5, 6, 7),
            target=(40, 41, 42, 43, 44, 45, 46, 47),
        ),
        _sample(
            phase="decode",
            source=(0, 1, 8, 9, 10, 11, 12, 13),
            target=(40, 41, 50, 51, 52, 53, 54, 55),
        ),
    )
    return TransitionPredictor.fit(
        samples, weighting="source_normalized", phase_mode="separate"
    )


def _identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        model_sha256="11" * 32,
        runtime_sha256="22" * 32,
        manifest_sha256="33" * 32,
        configuration_sha256="44" * 32,
    )


def test_runtime_artifact_is_deterministic_and_self_validating() -> None:
    first = build_runtime_artifact(_predictor(), _identity())
    second = build_runtime_artifact(_predictor(), _identity())

    assert first == second
    parsed = parse_runtime_artifact(first)
    assert parsed.format_version == 1
    assert parsed.source_layer == 23
    assert parsed.target_layer == 24
    assert parsed.expert_count == 128
    assert parsed.source_width == 8
    assert parsed.candidate_width == 12
    assert parsed.identity == _identity()
    assert parsed.payload_sha256 == sha256(parsed.payload).hexdigest()


def test_artifact_identity_has_stable_json_payload() -> None:
    assert artifact_identity_payload(_identity()) == {
        "configuration_sha256": "44" * 32,
        "manifest_sha256": "33" * 32,
        "model_sha256": "11" * 32,
        "runtime_sha256": "22" * 32,
    }


def test_runtime_artifact_reproduces_phase_specific_order_and_scores() -> None:
    artifact = parse_runtime_artifact(build_runtime_artifact(_predictor(), _identity()))
    source = (0, 1, 2, 3, 4, 5, 6, 7)

    prefill_ids, prefill_scores = predict_runtime_artifact(
        artifact, phase="prefill", source_expert_ids=source
    )
    decode_ids, decode_scores = predict_runtime_artifact(
        artifact, phase="decode", source_expert_ids=source
    )

    assert prefill_ids[:8] == (20, 21, 22, 23, 24, 25, 26, 27)
    assert decode_ids[:8] == (40, 41, 42, 43, 44, 45, 46, 47)
    assert len(prefill_ids) == len(prefill_scores) == 12
    assert len(decode_ids) == len(decode_scores) == 12
    assert all(
        left >= right
        for left, right in zip(prefill_scores, prefill_scores[1:])
    )


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, b"BROKEN!!", "magic"),
        (8, (2).to_bytes(4, "little"), "version"),
    ],
)
def test_runtime_artifact_rejects_invalid_header(
    offset: int, replacement: bytes, message: str
) -> None:
    artifact = bytearray(build_runtime_artifact(_predictor(), _identity()))
    artifact[offset : offset + len(replacement)] = replacement
    with pytest.raises(ValueError, match=message):
        parse_runtime_artifact(bytes(artifact))


def test_runtime_artifact_rejects_truncation_and_payload_corruption() -> None:
    artifact = build_runtime_artifact(_predictor(), _identity())
    with pytest.raises(ValueError, match="truncated"):
        parse_runtime_artifact(artifact[:-1])

    corrupted = bytearray(artifact)
    corrupted[-1] ^= 0x01
    with pytest.raises(ValueError, match="checksum"):
        parse_runtime_artifact(bytes(corrupted))


def test_fixture_output_preserves_candidate_order_and_scores(tmp_path: Path) -> None:
    artifact = parse_runtime_artifact(build_runtime_artifact(_predictor(), _identity()))
    output = tmp_path / "fixtures.json"
    write_parity_fixtures(
        artifact,
        (
            ("prefill", (0, 1, 2, 3, 4, 5, 6, 7)),
            ("decode", (0, 1, 2, 3, 4, 5, 6, 7)),
        ),
        output,
    )
    payload = output.read_text(encoding="utf-8")
    assert '"candidate_width": 12' in payload
    assert '"phase": "prefill"' in payload
    assert '"phase": "decode"' in payload
    assert payload == output.read_text(encoding="utf-8")
