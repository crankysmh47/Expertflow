from expertflow.predictor.dataset import PredictionSample
from expertflow.predictor.shadow import simulate_shadow


def _sample(token: int, target: tuple[int, ...]) -> PredictionSample:
    vector = tuple(1.0 if i == 0 else 0.0 for i in range(128))
    return PredictionSample("conversation", "validation", "domain", "decode", token, token, token,
                            0, 1, (0,), target, None, vector, None)


def _ranking(prefix: tuple[int, ...]) -> tuple[int, ...]:
    return prefix + tuple(i for i in range(128) if i not in prefix)


def test_shadow_accounts_for_useful_waste_uncovered_and_eviction_regret() -> None:
    samples = [_sample(0, (1,)), _sample(1, (1,))]
    rankings = [_ranking((1, 2)), _ranking((4, 5))]
    result = simulate_shadow(samples, rankings, width=2, capacity=2, expert_bytes=100)
    assert result["demand_count"] == 2
    assert result["reactive_demand_hits"] == 1
    assert result["useful_speculative_insertions"] == 1
    assert result["wasted_speculative_insertions"] == 3
    assert result["uncovered_blocking_misses"] == 1
    assert result["additional_speculative_evictions"] == 2
    assert result["eviction_regret"] == 1
    assert result["useful_predicted_bytes"] == 100
    assert result["wasted_predicted_bytes"] == 300
    assert result["late_predictions"] == 0
