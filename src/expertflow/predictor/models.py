"""Deterministic offline expert-ranking baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from expertflow.predictor.dataset import PredictionSample


def _complete(prefix: Iterable[int], scores: Counter[int] | None = None) -> tuple[int, ...]:
    ordered: list[int] = []
    seen: set[int] = set()
    for expert in prefix:
        if expert not in seen:
            ordered.append(expert)
            seen.add(expert)
    remaining = (scores or Counter())
    ordered.extend(sorted((expert for expert in range(128) if expert not in seen),
                          key=lambda expert: (-remaining[expert], expert)))
    return tuple(ordered)


class CopyPredictor:
    name = "b0_copy"

    def rank(self, sample: PredictionSample) -> tuple[int, ...]:
        return _complete(sample.source_expert_ids)


@dataclass(frozen=True, slots=True)
class FrequencyPredictor:
    counts: dict[int, Counter[int]]
    name: str = "b1_frequency"

    @classmethod
    def fit(cls, samples: Iterable[PredictionSample]) -> "FrequencyPredictor":
        counts: dict[int, Counter[int]] = defaultdict(Counter)
        for sample in samples:
            counts[sample.target_layer].update(sample.target_expert_ids)
        return cls(dict(counts))

    def rank(self, sample: PredictionSample) -> tuple[int, ...]:
        counts = self.counts.get(sample.target_layer, Counter())
        return tuple(sorted(range(128), key=lambda expert: (-counts[expert], expert)))


@dataclass(frozen=True, slots=True)
class TransitionPredictor:
    transitions: dict[int, dict[int, Counter[int]]]
    fallback: FrequencyPredictor
    name: str = "b2_transition"

    @classmethod
    def fit(cls, samples: Iterable[PredictionSample]) -> "TransitionPredictor":
        materialized = tuple(samples)
        transitions: dict[int, dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
        for sample in materialized:
            for source_expert in sample.source_expert_ids:
                transitions[sample.target_layer][source_expert].update(sample.target_expert_ids)
        frozen = {layer: dict(by_source) for layer, by_source in transitions.items()}
        return cls(frozen, FrequencyPredictor.fit(materialized))

    def rank(self, sample: PredictionSample) -> tuple[int, ...]:
        scores: Counter[int] = Counter()
        by_source = self.transitions.get(sample.target_layer, {})
        for source_expert in sample.source_expert_ids:
            scores.update(by_source.get(source_expert, Counter()))
        if not scores:
            return self.fallback.rank(sample)
        return tuple(sorted(range(128), key=lambda expert: (-scores[expert], expert)))
