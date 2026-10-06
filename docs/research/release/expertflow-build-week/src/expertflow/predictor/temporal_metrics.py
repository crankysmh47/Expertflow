"""Metrics for same-layer next-token expert predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from expertflow.predictor.temporal_dataset import TemporalSample


WIDTHS = (8, 12, 16)


def _summary(
    rows: Sequence[tuple[TemporalSample, tuple[int, ...]]],
) -> dict[str, float | int]:
    if not rows:
        return {
            "sample_count": 0,
            "recall_at_8": 0.0,
            "recall_at_12": 0.0,
            "recall_at_16": 0.0,
            "exact_set_match_at_8": 0.0,
        }
    hits = {width: 0 for width in WIDTHS}
    exact = 0
    demands = 0
    for sample, ranking in rows:
        if len(ranking) != 128 or set(ranking) != set(range(128)):
            raise ValueError("temporal ranking must be a permutation of 128 experts")
        target = set(sample.target_expert_ids)
        demands += len(target)
        for width in WIDTHS:
            hits[width] += len(target & set(ranking[:width]))
        exact += set(ranking[:8]) == target
    return {
        "sample_count": len(rows),
        **{f"recall_at_{width}": hits[width] / demands for width in WIDTHS},
        "exact_set_match_at_8": exact / len(rows),
    }


def evaluate_temporal_predictions(
    samples: Iterable[TemporalSample],
    rankings: Sequence[tuple[int, ...]],
) -> dict[str, object]:
    rows = tuple(zip(tuple(samples), rankings, strict=True))
    result: dict[str, object] = _summary(rows)
    groups: dict[str, dict[str, list[tuple[TemporalSample, tuple[int, ...]]]]] = {
        "per_conversation": defaultdict(list),
        "per_domain": defaultdict(list),
    }
    for sample, ranking in rows:
        groups["per_conversation"][sample.conversation_id].append((sample, ranking))
        groups["per_domain"][sample.domain].append((sample, ranking))
    for group_name, grouped in groups.items():
        result[group_name] = {
            key: _summary(value) for key, value in sorted(grouped.items())
        }
    return result

