from expertflow.predictor.dataset import PredictionSample
from expertflow.predictor.models import CopyPredictor, FrequencyPredictor, TransitionPredictor


def _sample(
    source: tuple[int, ...],
    target: tuple[int, ...],
    *,
    layer: int = 1,
    phase: str = "decode",
) -> PredictionSample:
    vector = tuple(1.0 if i in source else 0.0 for i in range(128))
    return PredictionSample("train", "train", "domain", phase, 0, 0, 1, layer - 1, layer,
                            source, target, None, vector, None)


def test_copy_preserves_source_order_then_completes_ranking() -> None:
    ranking = CopyPredictor().rank(_sample((7, 2), (10, 11)))
    assert ranking[:2] == (7, 2)
    assert len(ranking) == 128
    assert len(set(ranking)) == 128


def test_frequency_uses_training_counts_per_target_layer_with_stable_ties() -> None:
    model = FrequencyPredictor.fit([
        _sample((1,), (9, 8), layer=1),
        _sample((2,), (9, 7), layer=1),
        _sample((3,), (20, 21), layer=2),
    ])
    assert model.rank(_sample((4,), (0,), layer=1))[:4] == (9, 7, 8, 0)
    assert model.rank(_sample((4,), (0,), layer=2))[:3] == (20, 21, 0)


def test_transition_combines_all_source_experts_and_falls_back_deterministically() -> None:
    model = TransitionPredictor.fit([
        _sample((1, 2), (40, 41)),
        _sample((1, 3), (40, 42)),
        _sample((2, 3), (41, 42)),
    ])
    assert model.rank(_sample((1, 2), (0,)))[:3] == (40, 41, 42)
    assert model.rank(_sample((99,), (0,)))[:3] == (40, 41, 42)


def test_transition_source_normalization_changes_ranking() -> None:
    training = (
        [_sample((1,), (10,)) for _ in range(6)]
        + [_sample((1,), (11,)) for _ in range(4)]
        + [_sample((2,), (20,))]
    )
    raw = TransitionPredictor.fit(training, weighting="raw_count")
    normalized = TransitionPredictor.fit(training, weighting="source_normalized")
    query = _sample((1, 2), (0,))

    assert raw.rank(query)[0] == 10
    assert normalized.rank(query)[0] == 20


def test_transition_separate_phase_tables_do_not_mix_prefill_and_decode() -> None:
    model = TransitionPredictor.fit(
        [
            _sample((1,), (20,), phase="prefill"),
            _sample((1,), (30,), phase="decode"),
            _sample((1,), (30,), phase="decode"),
        ],
        phase_mode="separate",
    )

    assert model.rank(_sample((1,), (0,), phase="prefill"))[0] == 20
    assert model.rank(_sample((1,), (0,), phase="decode"))[0] == 30


def test_transition_support_excludes_fallback_completion() -> None:
    model = TransitionPredictor.fit([_sample((1,), (40,))])
    query = _sample((1,), (0,))

    assert model.has_support(query, 40)
    assert not model.has_support(query, 0)
