"""Leakage-safe same-layer next-token samples from canonical routing traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from expertflow.trace.io import load_router_events
from expertflow.trace.schema import RouterTraceEvent


SPLITS = ("train", "validation", "test")
TARGET_LAYER = 24
EXPERT_COUNT = 128
ROUTER_WIDTH = 8


@dataclass(frozen=True, slots=True)
class TemporalSample:
    conversation_id: str
    split: str
    domain: str
    request_id: str
    turn_index: int
    phase: str
    layer_id: int
    source_forward_id: int
    target_forward_id: int
    source_token_index: int
    target_token_index: int
    source_token_id: int
    target_token_id: int
    source_hook_order: int
    target_hook_order: int
    source_expert_ids: tuple[int, ...]
    target_expert_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemporalDataset:
    train: tuple[TemporalSample, ...]
    validation: tuple[TemporalSample, ...]
    test: tuple[TemporalSample, ...]
    conversation_ids: dict[str, tuple[str, ...]]


def _validate_router_set(event: RouterTraceEvent) -> None:
    if len(event.selected_expert_ids) != ROUTER_WIDTH:
        raise ValueError(
            f"layer-24 routing must contain exactly eight experts; "
            f"forward {event.forward_id} has {len(event.selected_expert_ids)}"
        )
    for expert in event.selected_expert_ids:
        if not 0 <= expert < EXPERT_COUNT:
            raise ValueError(f"expert ID {expert} is outside the canonical range 0..127")


def _sample(
    conversation: str,
    split: str,
    domain: str,
    source: RouterTraceEvent,
    target: RouterTraceEvent,
) -> TemporalSample:
    if source.request_id != target.request_id:
        raise ValueError(f"request changed inside decode sequence for {conversation}")
    if source.turn_index != target.turn_index:
        raise ValueError(f"turn changed inside decode sequence for {conversation}")
    if target.forward_id != source.forward_id + 1:
        raise ValueError(f"decode records do not have consecutive forward IDs for {conversation}")
    if target.token_index != source.token_index + 1:
        raise ValueError(f"decode records do not have consecutive token indices for {conversation}")
    if target.hook_order <= source.hook_order:
        raise ValueError(f"decode records do not have causal hook ordering for {conversation}")
    if source.phase != "decode" or target.phase != "decode":
        raise ValueError("temporal samples must remain decode-only")
    if source.layer_id != TARGET_LAYER or target.layer_id != TARGET_LAYER:
        raise ValueError("temporal samples must remain on layer 24")
    return TemporalSample(
        conversation_id=conversation,
        split=split,
        domain=domain,
        request_id=source.request_id,
        turn_index=source.turn_index,
        phase="decode",
        layer_id=TARGET_LAYER,
        source_forward_id=source.forward_id,
        target_forward_id=target.forward_id,
        source_token_index=source.token_index,
        target_token_index=target.token_index,
        source_token_id=source.token_id,
        target_token_id=target.token_id,
        source_hook_order=source.hook_order,
        target_hook_order=target.hook_order,
        source_expert_ids=source.selected_expert_ids,
        target_expert_ids=target.selected_expert_ids,
    )


def load_temporal_dataset(
    manifest_path: Path,
    *,
    expected_split_counts: dict[str, int] | None = None,
    expected_domain_counts: dict[str, dict[str, int]] | None = None,
    require_unique_prompt_hashes: bool = False,
    materialize_splits: set[str] | None = None,
) -> TemporalDataset:
    """Load strict decode-token pairs while leaving unrequested splits sealed."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_runtime") != "expertflow-canonical-observer-v1":
        raise ValueError("manifest is not from the canonical Observer v1 runtime")
    if manifest.get("trace_generation") != "trace_v2_canonical_segmented":
        raise ValueError("manifest is not from the accepted canonical trace generation")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest shards must be a non-empty array")

    materialized = set(SPLITS) if materialize_splits is None else set(materialize_splits)
    if not materialized or not materialized <= set(SPLITS):
        raise ValueError("materialized splits must be a non-empty subset of train/validation/test")

    split_by_conversation: dict[str, str] = {}
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_hashes: set[str] = set()
    shard_rows: list[tuple[str, str, str, Path]] = []
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("status") != "passed":
            raise ValueError("every temporal shard must be a passed object")
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
        prompt_hash = shard.get("prompt_sha256")
        if require_unique_prompt_hashes:
            if not isinstance(prompt_hash, str) or not prompt_hash:
                raise ValueError("prompt hash is required")
            if prompt_hash in prompt_hashes:
                raise ValueError(f"duplicate prompt hash {prompt_hash}")
            prompt_hashes.add(prompt_hash)
        split_by_conversation[conversation] = split
        domain_counts[split][domain] += 1
        if split in materialized:
            shard_rows.append((conversation, split, domain, Path(trace["path"])))

    expected = expected_split_counts or {"train": 60, "validation": 12, "test": 12}
    observed = Counter(split_by_conversation.values())
    if dict(observed) != expected:
        raise ValueError(f"frozen split counts {dict(observed)} do not match {expected}")
    if expected_domain_counts is not None:
        actual = {split: dict(counts) for split, counts in domain_counts.items()}
        if actual != expected_domain_counts:
            raise ValueError(f"frozen domain counts {actual} do not match {expected_domain_counts}")

    samples: dict[str, list[TemporalSample]] = {split: [] for split in SPLITS}
    for conversation, split, domain, trace_path in shard_rows:
        by_forward: dict[int, RouterTraceEvent] = {}
        for event in load_router_events(trace_path):
            if event.conversation_id != conversation:
                raise ValueError(f"conversation mismatch in {trace_path}")
            if event.phase != "decode" or event.layer_id != TARGET_LAYER:
                continue
            _validate_router_set(event)
            if event.forward_id in by_forward:
                raise ValueError(
                    f"duplicate layer-24 decode record for {conversation} forward {event.forward_id}"
                )
            by_forward[event.forward_id] = event
        if len(by_forward) < 2:
            raise ValueError(f"conversation {conversation} has fewer than two decode layer-24 records")
        ordered = sorted(by_forward.values(), key=lambda event: event.forward_id)
        for source, target in zip(ordered, ordered[1:], strict=False):
            samples[split].append(_sample(conversation, split, domain, source, target))

    identifiers = {
        split: tuple(sorted(
            conversation for conversation, value in split_by_conversation.items() if value == split
        ))
        for split in SPLITS
    }
    return TemporalDataset(
        train=tuple(samples["train"]),
        validation=tuple(samples["validation"]),
        test=tuple(samples["test"]),
        conversation_ids=identifiers,
    )
