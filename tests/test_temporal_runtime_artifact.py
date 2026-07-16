from __future__ import annotations

from collections import Counter

import pytest

from expertflow.predictor.temporal_dataset import TemporalSample
from expertflow.predictor.temporal_models import TemporalTransitionPredictor
from expertflow.predictor.temporal_runtime_artifact import (
    TemporalArtifactIdentity,
    build_temporal_runtime_artifact,
    parse_temporal_runtime_artifact,
    predict_temporal_runtime_artifact,
)


def _sample(source: tuple[int, ...], target: tuple[int, ...]) -> TemporalSample:
    return TemporalSample(
        "conversation", "train", "domain", "request", 0, "decode", 24,
        1, 2, 1, 2, 101, 102, 30, 60, source, target,
    )


def _identity() -> TemporalArtifactIdentity:
    return TemporalArtifactIdentity(*("ab" * 32, "cd" * 32, "ef" * 32, "12" * 32))


def test_round_trip_reproduces_frozen_combined_scorer() -> None:
    source = tuple(range(8))
    predictor = TemporalTransitionPredictor.fit((
        _sample(source, tuple(range(20, 28))),
        _sample(source, (20, 21, 30, 31, 32, 33, 34, 35)),
    ))
    artifact = parse_temporal_runtime_artifact(
        build_temporal_runtime_artifact(predictor, _identity())
    )
    session_before = Counter({0: 2, 20: 4})

    candidates, scores, session_after = predict_temporal_runtime_artifact(
        artifact,
        source_expert_ids=source,
        session_counts=session_before,
    )

    expected_counts = Counter(session_before)
    expected_counts.update(source)
    transition_scores = predictor.scores(_sample(source, tuple(range(8))))
    transition_max = max(transition_scores) or 1.0
    session_max = max(expected_counts.values())
    expected_scores = tuple(
        0.5 * transition_scores[expert] / transition_max
        + 0.4 * (expert in source)
        + 0.1 * expected_counts[expert] / session_max
        for expert in range(128)
    )
    expected = tuple(sorted(
        (expert for expert in range(128) if expected_scores[expert] > 0.0),
        key=lambda expert: (-expected_scores[expert], expert),
    )[:16])
    assert candidates == expected
    assert scores == tuple(expected_scores[expert] for expert in expected)
    assert session_after == expected_counts


def test_artifact_rejects_corruption_and_wrong_identity() -> None:
    predictor = TemporalTransitionPredictor.fit((
        _sample(tuple(range(8)), tuple(range(20, 28))),
    ))
    payload = bytearray(build_temporal_runtime_artifact(predictor, _identity()))
    payload[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        parse_temporal_runtime_artifact(bytes(payload))


def test_predictor_rejects_invalid_ids_and_requires_sixteen_supported() -> None:
    predictor = TemporalTransitionPredictor.fit((
        _sample(tuple(range(8)), tuple(range(8))),
    ))
    artifact = parse_temporal_runtime_artifact(
        build_temporal_runtime_artifact(predictor, _identity())
    )
    with pytest.raises(ValueError, match="eight unique"):
        predict_temporal_runtime_artifact(
            artifact,
            source_expert_ids=(0,) * 8,
            session_counts=Counter(),
        )

