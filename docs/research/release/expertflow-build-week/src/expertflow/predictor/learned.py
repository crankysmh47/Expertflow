"""Fixed small CPU multilabel predictors for the bounded pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Iterable

import torch
from torch import nn

from expertflow.predictor.dataset import PredictionSample


SEED = 20260716
FEATURE_COUNT = 287


def feature_vector(sample: PredictionSample) -> torch.Tensor:
    values = list(sample.source_vector)
    layers = [0.0] * 30
    if not 0 <= sample.target_layer < len(layers):
        raise ValueError("target layer is outside the fixed 30-layer contract")
    layers[sample.target_layer] = 1.0
    values.extend(layers)
    values.append(1.0 if sample.phase == "decode" else 0.0)
    values.extend(sample.previous_target_vector or (0.0,) * 128)
    if len(values) != FEATURE_COUNT:
        raise ValueError("feature contract is not 287 values")
    return torch.tensor(values, dtype=torch.float32)


def _target(sample: PredictionSample) -> torch.Tensor:
    values = torch.zeros(128, dtype=torch.float32)
    values[list(sample.target_expert_ids)] = 1.0
    return values


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


@dataclass(slots=True)
class _LearnedPredictor:
    module: nn.Module
    name: str
    seed: int = SEED

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters())

    def rank(self, sample: PredictionSample) -> tuple[int, ...]:
        self.module.eval()
        with torch.inference_mode():
            logits = self.module(feature_vector(sample)).tolist()
        return tuple(
            sorted(range(128), key=lambda expert: (-logits[expert], expert))
        )

    def latency_ns(
        self,
        sample: PredictionSample,
        *,
        repetitions: int = 200,
    ) -> list[int]:
        self.module.eval()
        values = feature_vector(sample)
        results: list[int] = []
        with torch.inference_mode():
            for _ in range(10):
                self.module(values)
            for _ in range(repetitions):
                start = perf_counter_ns()
                self.module(values)
                results.append(perf_counter_ns() - start)
        return results

    def save(self, path: Path) -> None:
        torch.save(
            {
                "schema_version": "1.0.0",
                "name": self.name,
                "seed": self.seed,
                "feature_count": FEATURE_COUNT,
                "parameter_count": self.parameter_count,
                "state_dict": self.module.state_dict(),
            },
            path,
        )


def _fit(
    module: nn.Module,
    samples: Iterable[PredictionSample],
    *,
    epochs: int,
    learning_rate: float,
    name: str,
) -> _LearnedPredictor:
    materialized = tuple(samples)
    if not materialized:
        raise ValueError("learned predictor requires training samples")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
    x = torch.stack([feature_vector(sample) for sample in materialized])
    y = torch.stack([_target(sample) for sample in materialized])
    positives = y.sum(dim=0)
    pos_weight = (
        (len(materialized) - positives) / positives.clamp_min(1.0)
    ).clamp(1.0, 64.0)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    module.train()
    for _ in range(epochs):
        for offset in range(0, len(materialized), 512):
            prediction = module(x[offset : offset + 512])
            loss = loss_function(prediction, y[offset : offset + 512])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return _LearnedPredictor(module=module.cpu(), name=name)


class LinearPredictor:
    @staticmethod
    def fit(
        samples: Iterable[PredictionSample],
        *,
        epochs: int = 8,
    ) -> _LearnedPredictor:
        return _fit(
            nn.Linear(FEATURE_COUNT, 128),
            samples,
            epochs=epochs,
            learning_rate=0.02,
            name="b3_linear",
        )


class SharedMlpPredictor:
    @staticmethod
    def fit(
        samples: Iterable[PredictionSample],
        *,
        epochs: int = 8,
    ) -> _LearnedPredictor:
        return _fit(
            _Mlp(),
            samples,
            epochs=epochs,
            learning_rate=0.005,
            name="b4_shared_mlp",
        )


def load_learned(path: Path) -> _LearnedPredictor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    name = payload["name"]
    if name == "b3_linear":
        module: nn.Module = nn.Linear(FEATURE_COUNT, 128)
    elif name == "b4_shared_mlp":
        module = _Mlp()
    else:
        raise ValueError(f"unsupported learned predictor artifact {name!r}")
    module.load_state_dict(payload["state_dict"])
    return _LearnedPredictor(
        module=module.cpu(),
        name=name,
        seed=int(payload["seed"]),
    )
