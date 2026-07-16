from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from expertflow.predictor.dataset import PredictionSample
from expertflow.predictor.learned import (
    LinearPredictor,
    SharedMlpPredictor,
    feature_vector,
)


def _sample(source: int, target: int) -> PredictionSample:
    source_ids = (source,)
    vector = tuple(1.0 if index == source else 0.0 for index in range(128))
    return PredictionSample(
        "train",
        "train",
        "domain",
        "decode",
        0,
        source,
        source,
        0,
        1,
        source_ids,
        (target,),
        None,
        vector,
        None,
    )


def test_feature_vector_has_fixed_contract() -> None:
    values = feature_vector(_sample(3, 9))
    assert values.shape == (287,)
    assert values[3] == 1.0
    assert values[128 + 1] == 1.0
    assert values[158] == 1.0


@pytest.mark.parametrize("model_type", [LinearPredictor, SharedMlpPredictor])
def test_fixed_cpu_models_are_deterministic_and_serializable(
    tmp_path: Path,
    model_type,
) -> None:
    samples = tuple(_sample(1, 9) for _ in range(24)) + tuple(
        _sample(2, 10) for _ in range(24)
    )
    first = model_type.fit(samples, epochs=4)
    second = model_type.fit(samples, epochs=4)
    assert first.rank(_sample(1, 9)) == second.rank(_sample(1, 9))
    assert first.parameter_count > 0
    path = tmp_path / "model.pt"
    first.save(path)
    assert path.stat().st_size > 0
    assert first.rank(_sample(1, 9))[0] in {9, 10}
