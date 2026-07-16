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
    transitions: dict[str, dict[int, dict[int, Counter[int]]]]
    fallback: FrequencyPredictor
    weighting: str = "raw_count"
    phase_mode: str = "pooled"
    name: str = "b2_transition"

    @classmethod
    def fit(
        cls,
        samples: Iterable[PredictionSample],
        *,
        weighting: str = "raw_count",
        phase_mode: str = "pooled",
    ) -> "TransitionPredictor":
        if weighting not in {"raw_count", "source_normalized"}:
            raise ValueError("transition weighting must be raw_count or source_normalized")
        if phase_mode not in {"pooled", "separate"}:
            raise ValueError("transition phase mode must be pooled or separate")
        materialized = tuple(samples)
        transitions: dict[str, dict[int, dict[int, Counter[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(Counter))
        )
        for sample in materialized:
            phase = sample.phase if phase_mode == "separate" else "*"
            for source_expert in sample.source_expert_ids:
                transitions[phase][sample.target_layer][source_expert].update(sample.target_expert_ids)
        frozen = {
            phase: {
                layer: dict(by_source)
                for layer, by_source in by_layer.items()
            }
            for phase, by_layer in transitions.items()
        }
        return cls(frozen, FrequencyPredictor.fit(materialized), weighting, phase_mode)

    def _scores(self, sample: PredictionSample) -> Counter[int]:
        scores: Counter[int] = Counter()
        phase = sample.phase if self.phase_mode == "separate" else "*"
        by_source = self.transitions.get(phase, {}).get(sample.target_layer, {})
        for source_expert in sample.source_expert_ids:
            source_counts = by_source.get(source_expert, Counter())
            if self.weighting == "raw_count":
                scores.update(source_counts)
            else:
                total = source_counts.total()
                if total:
                    for target_expert, count in source_counts.items():
                        scores[target_expert] += count / total
        return scores

    def rank(self, sample: PredictionSample) -> tuple[int, ...]:
        scores = self._scores(sample)
        if not scores:
            return self.fallback.rank(sample)
        return tuple(sorted(range(128), key=lambda expert: (-scores[expert], expert)))

    def has_support(self, sample: PredictionSample, expert_id: int) -> bool:
        return self._scores(sample)[expert_id] > 0
