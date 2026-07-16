from expertflow.predictor.temporal_dataset import TemporalSample
from expertflow.predictor.temporal_shadow import simulate_temporal_shadow


def _sample(source: tuple[int, ...], target: tuple[int, ...], step: int) -> TemporalSample:
    return TemporalSample(
        "a", "validation", "domain", "request", 0, "decode", 24,
        step, step + 1, step, step + 1, 100 + step, 101 + step,
        step * 30, (step + 1) * 30, source, target,
    )


def test_shadow_applies_next_token_predictions_before_target_demand() -> None:
    samples = (
        _sample((0, 1), (2, 3), 0),
        _sample((2, 3), (4, 5), 1),
    )
    rankings = (
        (2, 3) + tuple(i for i in range(128) if i not in {2, 3}),
        (4, 99) + tuple(i for i in range(128) if i not in {4, 99}),
    )
    result = simulate_temporal_shadow(
        samples, rankings, width=2, capacity=4, expert_bytes=10
    )
    assert result["reactive_blocking_misses"] == 4
    assert result["ready_improvement_over_reactive"] == 3
    assert result["useful_speculative_insertions"] == 3
    assert result["wasted_speculative_insertions"] == 1
    assert result["useful_predicted_bytes"] == 30
    assert result["wasted_predicted_bytes"] == 10


def test_shadow_emits_zero_valued_selection_metrics() -> None:
    samples = (_sample((0, 1), (0, 1), 0),)
    rankings = ((0, 1) + tuple(range(2, 128)),)

    result = simulate_temporal_shadow(
        samples, rankings, width=2, capacity=4, expert_bytes=10
    )

    assert result["eviction_regret"] == 0
    assert result["wasted_predicted_bytes"] == 0
    assert result["additional_speculative_evictions"] == 0
