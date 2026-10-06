"""Metrics for unordered next-layer expert sets."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Protocol

from expertflow.predictor.dataset import PredictionSample


class Ranker(Protocol):
    def rank(self, sample: PredictionSample) -> tuple[int, ...]: ...


def _summary(rows: list[tuple[PredictionSample, tuple[int, ...]]]) -> dict[str, float | int]:
    if not rows:
        return {
            "sample_count": 0, "recall_at_8": 0.0, "recall_at_12": 0.0,
            "recall_at_16": 0.0, "mean_overlap_at_8": 0.0, "exact_set_match_at_8": 0.0,
        }
    hits = {width: 0 for width in (8, 12, 16)}
    exact = 0
    for sample, ranking in rows:
        if len(ranking) != 128 or len(set(ranking)) != 128 or set(ranking) != set(range(128)):
            raise ValueError("predictor ranking must be a permutation of 128 experts")
        target = set(sample.target_expert_ids)
        for width in hits:
            hits[width] += len(target & set(ranking[:width]))
        exact += set(ranking[:8]) == target
    demands = sum(len(sample.target_expert_ids) for sample, _ in rows)
    count = len(rows)
    return {
        "sample_count": count,
        "recall_at_8": hits[8] / demands,
        "recall_at_12": hits[12] / demands,
        "recall_at_16": hits[16] / demands,
        "mean_overlap_at_8": hits[8] / count,
        "exact_set_match_at_8": exact / count,
    }


def evaluate_predictions(samples: Iterable[PredictionSample], predictor: Ranker) -> dict[str, object]:
    rows = [(sample, predictor.rank(sample)) for sample in samples]
    result: dict[str, object] = _summary(rows)
    groups: dict[str, dict[str, list[tuple[PredictionSample, tuple[int, ...]]]]] = {
        "per_layer": defaultdict(list),
        "per_phase": defaultdict(list),
        "per_conversation": defaultdict(list),
        "per_domain": defaultdict(list),
    }
    for sample, ranking in rows:
        groups["per_layer"][str(sample.target_layer)].append((sample, ranking))
        groups["per_phase"][sample.phase].append((sample, ranking))
        groups["per_conversation"][sample.conversation_id].append((sample, ranking))
        groups["per_domain"][sample.domain].append((sample, ranking))
    for name, grouped in groups.items():
        result[name] = {key: _summary(value) for key, value in sorted(grouped.items())}
    return result
