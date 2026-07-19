"""Bounded deterministic policies for same-layer next-token expert prediction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from expertflow.predictor.temporal_dataset import EXPERT_COUNT, TemporalSample


def _complete(prefix: Iterable[int], scores: Sequence[float] | Counter[int] | None = None) -> tuple[int, ...]:
    ordered: list[int] = []
    seen: set[int] = set()
    for expert in prefix:
        if expert not in seen:
            ordered.append(expert)
            seen.add(expert)
    if scores is None:
        ordered.extend(expert for expert in range(EXPERT_COUNT) if expert not in seen)
    else:
        ordered.extend(sorted(
            (expert for expert in range(EXPERT_COUNT) if expert not in seen),
            key=lambda expert: (-scores[expert], expert),
        ))
    return tuple(ordered)


class TemporalRanker(Protocol):
    name: str

    def rank(self, sample: TemporalSample, session_counts: Counter[int]) -> tuple[int, ...]: ...


class TemporalCopyPredictor:
    name = "t0.0_copy"

    def rank(self, sample: TemporalSample, session_counts: Counter[int]) -> tuple[int, ...]:
        return _complete(sample.source_expert_ids)


class TemporalSessionFrequencyPredictor:
    name = "t0.1_session_frequency"

    def rank(self, sample: TemporalSample, session_counts: Counter[int]) -> tuple[int, ...]:
        current_order = {expert: index for index, expert in enumerate(sample.source_expert_ids)}
        return tuple(sorted(
            range(EXPERT_COUNT),
            key=lambda expert: (
                -session_counts[expert],
                current_order.get(expert, EXPERT_COUNT),
                expert,
            ),
        ))


@dataclass(frozen=True, slots=True)
class TemporalTransitionPredictor:
    transitions: dict[int, Counter[int]]
    target_frequency: Counter[int]
    name: str = "t0.2_transition"

    @classmethod
    def fit(cls, samples: Iterable[TemporalSample]) -> "TemporalTransitionPredictor":
        rows: dict[int, Counter[int]] = defaultdict(Counter)
        target_frequency: Counter[int] = Counter()
        for sample in samples:
            target_frequency.update(sample.target_expert_ids)
            for source in sample.source_expert_ids:
                rows[source].update(sample.target_expert_ids)
        return cls(dict(rows), target_frequency)

    def scores(self, sample: TemporalSample) -> tuple[float, ...]:
        scores = [0.0] * EXPERT_COUNT
        for source in sample.source_expert_ids:
            row = self.transitions.get(source, Counter())
            total = row.total()
            if total:
                for target, count in row.items():
                    scores[target] += count / total
        return tuple(scores)

    def rank(self, sample: TemporalSample, session_counts: Counter[int]) -> tuple[int, ...]:
        scores = self.scores(sample)
        return tuple(sorted(
            range(EXPERT_COUNT),
            key=lambda expert: (-scores[expert], -self.target_frequency[expert], expert),
        ))


@dataclass(frozen=True, slots=True)
class TemporalCombinedPredictor:
    transition: TemporalTransitionPredictor
    weights: tuple[float, float, float]
    name: str = "t0.3_combined"

    def __post_init__(self) -> None:
        if any(weight < 0 for weight in self.weights) or abs(sum(self.weights) - 1.0) > 1e-12:
            raise ValueError("combined temporal weights must be non-negative and sum to one")

    def rank(self, sample: TemporalSample, session_counts: Counter[int]) -> tuple[int, ...]:
        transition_scores = self.transition.scores(sample)
        transition_scale = max(transition_scores) or 1.0
        session_scale = max(session_counts.values(), default=0) or 1
        current = set(sample.source_expert_ids)
        transition_weight, retention_weight, session_weight = self.weights
        scores = [
            transition_weight * transition_scores[expert] / transition_scale
            + retention_weight * (expert in current)
            + session_weight * session_counts[expert] / session_scale
            for expert in range(EXPERT_COUNT)
        ]
        return tuple(sorted(range(EXPERT_COUNT), key=lambda expert: (-scores[expert], expert)))


def rank_temporal_samples(
    samples: Iterable[TemporalSample],
    predictor: TemporalRanker,
) -> tuple[tuple[int, ...], ...]:
    """Rank in causal conversation order with resettable session state."""

    current_conversation: str | None = None
    session_counts: Counter[int] = Counter()
    rankings: list[tuple[int, ...]] = []
    previous: TemporalSample | None = None
    for sample in samples:
        if sample.conversation_id != current_conversation:
            current_conversation = sample.conversation_id
            session_counts.clear()
            previous = None
        elif previous is not None and sample.source_expert_ids != previous.target_expert_ids:
            raise ValueError("temporal samples are not a continuous causal conversation sequence")
        session_counts.update(sample.source_expert_ids)
        ranking = predictor.rank(sample, session_counts)
        if len(ranking) != EXPERT_COUNT or set(ranking) != set(range(EXPERT_COUNT)):
            raise ValueError("temporal predictor ranking must be a permutation of 128 experts")
        rankings.append(ranking)
        previous = sample
    return tuple(rankings)

