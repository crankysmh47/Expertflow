"""Leakage-safe adjacent-layer samples from canonical routing traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from expertflow.trace.io import load_router_events
from expertflow.trace.schema import RouterTraceEvent


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class PredictionSample:
    conversation_id: str
    split: str
    domain: str
    phase: str
    forward_id: int
    token_index: int
    token_id: int
    source_layer: int
    target_layer: int
    source_expert_ids: tuple[int, ...]
    target_expert_ids: tuple[int, ...]
    source_expert_weights: tuple[float, ...] | None
    source_vector: tuple[float, ...]
    previous_target_vector: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class PilotDataset:
    train: tuple[PredictionSample, ...]
    validation: tuple[PredictionSample, ...]
    test: tuple[PredictionSample, ...]
    conversation_ids: dict[str, tuple[str, ...]]


def _vector(ids: tuple[int, ...], weights: tuple[float, ...] | None = None) -> tuple[float, ...]:
    result = [0.0] * 128
    values = weights if weights is not None else (1.0,) * len(ids)
    for expert_id, value in zip(ids, values, strict=True):
        if not 0 <= expert_id < 128:
            raise ValueError(f"expert ID {expert_id} is outside the canonical 128-expert range")
        result[expert_id] = float(value)
    return tuple(result)


def _execution_key(event: RouterTraceEvent) -> tuple[int, int, int]:
    return event.forward_id, event.token_index, event.token_id


def load_pilot_dataset(
    manifest_path: Path,
    *,
    expected_split_counts: dict[str, int] | None = None,
) -> PilotDataset:
    """Load a canonical manifest and fail closed on any ambiguous adjacent-layer join."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_runtime") != "expertflow-canonical-observer-v1":
        raise ValueError("manifest is not from the canonical Observer v1 runtime")
    if manifest.get("trace_generation") != "trace_v2_canonical_segmented":
        raise ValueError("manifest is not from the accepted canonical trace generation")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest shards must be a non-empty array")

    split_by_conversation: dict[str, str] = {}
    shard_rows: list[tuple[str, str, str, Path]] = []
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("status") != "passed":
            raise ValueError("every predictor shard must be a passed object")
        conversation = shard.get("conversation_id")
        split = shard.get("split")
        domain = shard.get("domain")
        trace = shard.get("trace")
        if not isinstance(conversation, str) or split not in SPLITS or not isinstance(domain, str):
            raise ValueError("shard conversation, split, and domain are invalid")
        if conversation in split_by_conversation:
            if split_by_conversation[conversation] != split:
                raise ValueError(f"conversation {conversation} appears in multiple splits")
            raise ValueError(f"duplicate shard for conversation {conversation}")
        if not isinstance(trace, dict) or not isinstance(trace.get("path"), str):
            raise ValueError("shard trace path is invalid")
        split_by_conversation[conversation] = split
        shard_rows.append((conversation, split, domain, Path(trace["path"])))

    expected = expected_split_counts or {"train": 7, "validation": 4, "test": 3}
    observed = Counter(split_by_conversation.values())
    if dict(observed) != expected:
        raise ValueError(f"frozen split counts {dict(observed)} do not match {expected}")

    grouped_by_shard: list[tuple[str, str, str, dict[tuple[int, int, int], dict[int, RouterTraceEvent]]]] = []
    expected_layers: set[int] = set()
    for conversation, split, domain, trace_path in shard_rows:
        groups: dict[tuple[int, int, int], dict[int, RouterTraceEvent]] = defaultdict(dict)
        for event in load_router_events(trace_path):
            if event.conversation_id != conversation:
                raise ValueError(f"conversation mismatch in {trace_path}")
            key = _execution_key(event)
            if event.layer_id in groups[key]:
                raise ValueError(f"duplicate layer event for {conversation} execution {key}")
            groups[key][event.layer_id] = event
            expected_layers.add(event.layer_id)
        grouped_by_shard.append((conversation, split, domain, groups))

    ordered_layers = tuple(sorted(expected_layers))
    if len(ordered_layers) < 2 or any(right != left + 1 for left, right in zip(ordered_layers, ordered_layers[1:])):
        raise ValueError("canonical MoE layers must form one consecutive sequence")

    samples: dict[str, list[PredictionSample]] = {split: [] for split in SPLITS}
    for conversation, split, domain, groups in grouped_by_shard:
        previous_by_target: dict[int, tuple[float, ...]] = {}
        ordered_groups = sorted(
            groups.items(),
            key=lambda item: min(event.hook_order for event in item[1].values()),
        )
        for key, layer_events in ordered_groups:
            if tuple(sorted(layer_events)) != ordered_layers:
                raise ValueError(f"incomplete layer sequence for {conversation} execution {key}")
            phases = {event.phase for event in layer_events.values()}
            if len(phases) != 1:
                raise ValueError(f"phase mismatch for {conversation} execution {key}")
            for source_layer, target_layer in zip(ordered_layers, ordered_layers[1:]):
                source = layer_events[source_layer]
                target = layer_events[target_layer]
                if target_layer != source_layer + 1:
                    raise ValueError("target layer is not the true next MoE layer")
                samples[split].append(PredictionSample(
                    conversation_id=conversation,
                    split=split,
                    domain=domain,
                    phase=source.phase,
                    forward_id=source.forward_id,
                    token_index=source.token_index,
                    token_id=source.token_id,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    source_expert_ids=source.selected_expert_ids,
                    target_expert_ids=target.selected_expert_ids,
                    source_expert_weights=source.selected_expert_weights,
                    source_vector=_vector(source.selected_expert_ids, source.selected_expert_weights),
                    previous_target_vector=previous_by_target.get(target_layer),
                ))
            previous_by_target = {
                layer: _vector(event.selected_expert_ids)
                for layer, event in layer_events.items()
            }

    identifiers = {
        split: tuple(sorted(conversation for conversation, value in split_by_conversation.items() if value == split))
        for split in SPLITS
    }
    return PilotDataset(
        train=tuple(samples["train"]),
        validation=tuple(samples["validation"]),
        test=tuple(samples["test"]),
        conversation_ids=identifiers,
    )
