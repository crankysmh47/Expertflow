from __future__ import annotations

import pytest

from expertflow.quality.analysis import evaluate_quality_gate


def _result(**overrides):
    result = {
        "perplexity": 10.0,
        "mmlu_correct": 80,
        "mmlu_total": 100,
        "repeated_4gram_rate": 0.10,
        "distinct_2": 0.70,
        "deterministic": True,
        "runtime_clean": True,
    }
    result.update(overrides)
    return result


def test_quality_gate_applies_frozen_thresholds():
    result = evaluate_quality_gate(
        _result(),
        _result(
            perplexity=10.04,
            mmlu_correct=79,
            repeated_4gram_rate=0.14,
            distinct_2=0.66,
        ),
    )

    assert result["pass"] is True
    assert result["perplexity_relative_change"] == pytest.approx(0.004)
    assert result["mmlu_accuracy_delta_pp"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (_result(perplexity=10.051), "perplexity"),
        (_result(mmlu_correct=78), "mmlu_accuracy"),
        (_result(repeated_4gram_rate=0.151), "repeated_4gram"),
        (_result(distinct_2=0.649), "distinct_2"),
        (_result(deterministic=False), "candidate_determinism"),
        (_result(runtime_clean=False), "runtime_cleanliness"),
    ],
)
def test_quality_gate_names_each_failure(candidate, reason):
    result = evaluate_quality_gate(_result(), candidate)

    assert result["pass"] is False
    assert reason in result["failed_gates"]


def test_quality_gate_rejects_incomplete_scores():
    candidate = _result()
    candidate.pop("perplexity")

    with pytest.raises(ValueError, match="candidate missing"):
        evaluate_quality_gate(_result(), candidate)
