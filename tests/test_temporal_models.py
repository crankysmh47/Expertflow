from collections import Counter

import pytest

from expertflow.predictor.temporal_dataset import TemporalSample
from expertflow.predictor.temporal_models import (
    TemporalCombinedPredictor,
    TemporalCopyPredictor,
    TemporalSessionFrequencyPredictor,
    TemporalTransitionPredictor,
    rank_temporal_samples,
)


def _sample(
    conversation: str,
    source: tuple[int, ...],
    target: tuple[int, ...],
    step: int,
) -> TemporalSample:
    return TemporalSample(
        conversation, "train", "domain", "request", 0, "decode", 24,
        step, step + 1, step, step + 1, 100 + step, 101 + step,
        step * 30, (step + 1) * 30, source, target,
    )


def test_copy_preserves_current_router_order() -> None:
    sample = _sample("a", (7, 3, 9, 1, 5, 2, 8, 4), tuple(range(8)), 0)
    ranking = rank_temporal_samples((sample,), TemporalCopyPredictor())[0]
    assert ranking[:8] == sample.source_expert_ids
    assert len(ranking) == len(set(ranking)) == 128


def test_session_frequency_is_causal_and_resets_per_conversation() -> None:
    samples = (
        _sample(
            "a",
            (5, 1, 2, 3, 4, 6, 7, 8),
            (5, 9, 10, 11, 12, 13, 14, 15),
            0,
        ),
        _sample("a", (5, 9, 10, 11, 12, 13, 14, 15), tuple(range(8)), 1),
        _sample("b", (20, 21, 22, 23, 24, 25, 26, 27), tuple(range(8)), 0),
    )
    rankings = rank_temporal_samples(samples, TemporalSessionFrequencyPredictor())
    assert rankings[1][0] == 5
    assert rankings[2][:8] == samples[2].source_expert_ids


def test_transition_is_source_normalized_and_training_only() -> None:
    train = (
        _sample("a", (0, 1, 2, 3, 4, 5, 6, 7), (20, 21, 22, 23, 24, 25, 26, 27), 0),
        _sample("a", (0, 8, 9, 10, 11, 12, 13, 14), (20, 30, 31, 32, 33, 34, 35, 36), 1),
    )
    predictor = TemporalTransitionPredictor.fit(train)
    query = _sample("q", (0, 40, 41, 42, 43, 44, 45, 46), tuple(range(8)), 0)
    scores = predictor.scores(query)
    assert scores[20] == pytest.approx(0.125)
    assert scores[21] == pytest.approx(0.0625)
    assert predictor.rank(query, Counter())[0] == 20
    assert 127 not in predictor.target_frequency


def test_combined_scorer_uses_fixed_valid_weights() -> None:
    transition = TemporalTransitionPredictor.fit((
        _sample("a", tuple(range(8)), tuple(range(20, 28)), 0),
    ))
    predictor = TemporalCombinedPredictor(transition, (0.5, 0.25, 0.25))
    query = _sample("q", tuple(range(8)), tuple(range(8)), 0)
    ranking = predictor.rank(query, Counter({99: 10}))
    assert len(ranking) == len(set(ranking)) == 128
    with pytest.raises(ValueError, match="sum to one"):
        TemporalCombinedPredictor(transition, (0.5, 0.5, 0.5))
