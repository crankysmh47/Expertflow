from expertflow.predictor.temporal_dataset import TemporalSample
from expertflow.predictor.temporal_metrics import evaluate_temporal_predictions


def _sample(conversation: str, target: tuple[int, ...]) -> TemporalSample:
    return TemporalSample(
        conversation, "validation", "domain-a", "request", 0, "decode", 24,
        0, 1, 0, 1, 100, 101, 0, 30, tuple(range(8)), target,
    )


def test_metrics_report_widths_conversations_and_domains() -> None:
    samples = (
        _sample("a", tuple(range(8))),
        _sample("b", tuple(range(4, 12))),
    )
    rankings = (tuple(range(128)), tuple(range(128)))
    result = evaluate_temporal_predictions(samples, rankings)
    assert result["sample_count"] == 2
    assert result["recall_at_8"] == 0.75
    assert result["recall_at_12"] == 1.0
    assert result["exact_set_match_at_8"] == 0.5
    assert result["per_conversation"]["b"]["recall_at_8"] == 0.5
    assert result["per_domain"]["domain-a"]["sample_count"] == 2

