from expertflow.predictor.dataset import PredictionSample
from expertflow.predictor.metrics import evaluate_predictions


def _sample(conversation: str, phase: str, layer: int, target: tuple[int, ...]) -> PredictionSample:
    source = tuple(range(8))
    vector = tuple(1.0 if i in source else 0.0 for i in range(128))
    return PredictionSample(conversation, "validation", "domain", phase, 0, 0, 1, layer - 1, layer,
                            source, target, None, vector, None)


class Fixed:
    def rank(self, sample: PredictionSample) -> tuple[int, ...]:
        return tuple(range(128))


def test_metrics_include_widths_and_breakdowns() -> None:
    samples = [
        _sample("a", "prefill", 1, tuple(range(8))),
        _sample("b", "decode", 2, tuple(range(4, 12))),
    ]
    result = evaluate_predictions(samples, Fixed())
    assert result["sample_count"] == 2
    assert result["recall_at_8"] == 0.75
    assert result["recall_at_12"] == 1.0
    assert result["recall_at_16"] == 1.0
    assert result["mean_overlap_at_8"] == 6.0
    assert result["exact_set_match_at_8"] == 0.5
    assert result["per_layer"]["1"]["recall_at_8"] == 1.0
    assert result["per_phase"]["decode"]["recall_at_8"] == 0.5
    assert result["per_conversation"]["b"]["mean_overlap_at_8"] == 4.0
    assert result["per_domain"]["domain"]["sample_count"] == 2
