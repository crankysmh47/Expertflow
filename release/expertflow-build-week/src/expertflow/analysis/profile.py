"""Deterministic locality summaries over canonical router events."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from expertflow.trace.schema import RouterTraceEvent


@dataclass(frozen=True, slots=True)
class LayerRoutingProfile:
    """Measured routing-locality summary for one MoE layer."""

    layer_id: int
    event_count: int
    selection_count: int
    unique_expert_count: int
    top_experts: tuple[tuple[int, int], ...]
    static_hit_rates: tuple[tuple[int, float], ...]
    adjacent_reuse_rate: float | None
    mean_reuse_distance_tokens: float | None


@dataclass(frozen=True, slots=True)
class RoutingProfile:
    """Measured profile spanning all observed MoE layers."""

    total_events: int
    layers: tuple[LayerRoutingProfile, ...]


@dataclass(slots=True)
class _LayerAccumulator:
    event_count: int
    expert_counts: Counter[int]
    adjacent_hits: int
    adjacent_opportunities: int
    reuse_distance_sum: int
    reuse_distance_count: int


def summarize_routing(
    events: Iterable[RouterTraceEvent],
    *,
    static_budgets: tuple[int, ...] = (1, 2, 4, 8),
) -> RoutingProfile:
    """Compute concentration and token-locality metrics without timing estimates."""

    budgets = tuple(sorted(set(static_budgets)))
    if any(budget <= 0 for budget in budgets):
        raise ValueError("static budgets must be positive")

    layers: defaultdict[int, _LayerAccumulator] = defaultdict(
        lambda: _LayerAccumulator(0, Counter(), 0, 0, 0, 0)
    )
    previous_selection: dict[tuple[str, int], frozenset[int]] = {}
    previous_token: dict[tuple[str, int, int], int] = {}
    total_events = 0

    for event in events:
        total_events += 1
        layer = layers[event.layer_id]
        layer.event_count += 1
        layer.expert_counts.update(event.selected_expert_ids)

        request_layer = (event.request_id, event.layer_id)
        prior_selection = previous_selection.get(request_layer)
        if prior_selection is not None:
            layer.adjacent_opportunities += len(event.selected_expert_ids)
            layer.adjacent_hits += len(
                prior_selection.intersection(event.selected_expert_ids)
            )
        previous_selection[request_layer] = frozenset(event.selected_expert_ids)

        for expert_id in event.selected_expert_ids:
            expert_key = (event.request_id, event.layer_id, expert_id)
            prior_token = previous_token.get(expert_key)
            if prior_token is not None:
                distance = event.token_index - prior_token
                if distance < 0:
                    raise ValueError(
                        "token_index moved backwards within a request and layer"
                    )
                layer.reuse_distance_sum += distance
                layer.reuse_distance_count += 1
            previous_token[expert_key] = event.token_index

    summaries: list[LayerRoutingProfile] = []
    for layer_id, accumulator in sorted(layers.items()):
        top_experts = tuple(
            sorted(
                accumulator.expert_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        selection_count = accumulator.expert_counts.total()
        static_hit_rates = tuple(
            (
                budget,
                sum(count for _, count in top_experts[:budget]) / selection_count,
            )
            for budget in budgets
        )
        adjacent_reuse_rate = (
            accumulator.adjacent_hits / accumulator.adjacent_opportunities
            if accumulator.adjacent_opportunities
            else None
        )
        mean_reuse_distance = (
            accumulator.reuse_distance_sum / accumulator.reuse_distance_count
            if accumulator.reuse_distance_count
            else None
        )
        summaries.append(
            LayerRoutingProfile(
                layer_id=layer_id,
                event_count=accumulator.event_count,
                selection_count=selection_count,
                unique_expert_count=len(accumulator.expert_counts),
                top_experts=top_experts,
                static_hit_rates=static_hit_rates,
                adjacent_reuse_rate=adjacent_reuse_rate,
                mean_reuse_distance_tokens=mean_reuse_distance,
            )
        )

    return RoutingProfile(total_events=total_events, layers=tuple(summaries))
