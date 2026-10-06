"""Per-conversation and per-domain held-out cache-policy evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from expertflow.analysis.capacity_curve import build_held_out_capacity_curve
from expertflow.trace.io import load_router_events
from expertflow.trace.schema import RouterTraceEvent


@dataclass(frozen=True)
class EvaluationTrace:
    """One untouched validation or test conversation trace."""

    conversation_id: str
    split: str
    domain: str
    source_trace: str
    events: tuple[RouterTraceEvent, ...]


def load_collection_breakdown_inputs(
    collection_manifest: Path,
    *,
    phase: str,
    evaluation_phase: str | None = None,
    max_layer: int | None,
    exclude_failed: bool = False,
) -> tuple[
    tuple[RouterTraceEvent, ...],
    tuple[EvaluationTrace, ...],
    tuple[dict[str, object], ...],
]:
    """Load complete, parity-passed shards without crossing split boundaries."""

    selected_evaluation_phase = evaluation_phase or phase
    if phase not in {"prefill", "decode"} or selected_evaluation_phase not in {
        "prefill",
        "decode",
    }:
        raise ValueError("training and evaluation phases must be prefill or decode")
    if max_layer is not None and max_layer < 0:
        raise ValueError("max_layer must be non-negative")
    try:
        payload = json.loads(collection_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read collection manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("collection manifest must be an object")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        conversation_count = summary.get("conversation_count")
        passed_count = summary.get("passed")
        failed_count = summary.get("failed")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (conversation_count, passed_count, failed_count)
        ):
            raise ValueError("collection summary counts must be integers")
        if passed_count + failed_count != conversation_count:
            raise ValueError("collection manifest is incomplete")
        if failed_count and not exclude_failed:
            raise ValueError("collection manifest has failed shards")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("collection manifest shards must be a non-empty array")

    training: list[RouterTraceEvent] = []
    evaluations: list[EvaluationTrace] = []
    excluded: list[dict[str, object]] = []
    for index, raw_shard in enumerate(shards):
        if not isinstance(raw_shard, dict):
            raise ValueError(f"shards[{index}] must be an object")
        attempts = raw_shard.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"shards[{index}] has no collection attempt")
        status = raw_shard.get("latest_status")
        if status != "passed":
            if not exclude_failed:
                raise ValueError(f"shards[{index}] is not parity-passed")
            conversation_id = raw_shard.get("conversation_id")
            split = raw_shard.get("split")
            domain = raw_shard.get("domain")
            if not all(
                isinstance(value, str)
                for value in (conversation_id, split, domain, status)
            ):
                raise ValueError(f"shards[{index}] identity is invalid")
            excluded.append(
                {
                    "conversation_id": conversation_id,
                    "split": split,
                    "domain": domain,
                    "latest_status": status,
                    "attempt_count": len(attempts),
                }
            )
            continue
        latest = attempts[-1]
        try:
            trace_value = latest["trace"]["artifact"]["path"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"shards[{index}] has no trace artifact path"
            ) from error
        if not isinstance(trace_value, str):
            raise ValueError(f"shards[{index}] trace path must be a string")
        trace_path = Path(trace_value).resolve()
        split = raw_shard.get("split")
        selected_phase = (
            phase if split == "train" else selected_evaluation_phase
        )
        events = tuple(
            event
            for event in load_router_events(trace_path)
            if event.phase == selected_phase
            and (max_layer is None or event.layer_id <= max_layer)
        )
        if not events:
            raise ValueError(f"shards[{index}] selection produced no events")
        if split == "train":
            training.extend(events)
            continue
        if split not in {"validation", "test"}:
            raise ValueError(f"shards[{index}] has unsupported split")
        conversation_id = raw_shard.get("conversation_id")
        domain = raw_shard.get("domain")
        if not isinstance(conversation_id, str) or not isinstance(domain, str):
            raise ValueError(f"shards[{index}] identity is invalid")
        evaluations.append(
            EvaluationTrace(
                conversation_id=conversation_id,
                split=split,
                domain=domain,
                source_trace=str(trace_path),
                events=events,
            )
        )
    if not training or not evaluations:
        raise ValueError("collection must contain training and held-out shards")
    return tuple(training), tuple(evaluations), tuple(excluded)


def _token_count(events: tuple[RouterTraceEvent, ...]) -> int:
    return len({(event.request_id, event.forward_id) for event in events})


def _policy_metrics(
    policy: dict[str, object],
    *,
    token_count: int,
    slot_bytes: int,
    expert_transfer_ms: float,
) -> dict[str, object]:
    demand_count = int(policy["demand_count"])
    hit_count = int(policy["hit_count"])
    miss_count = int(policy["miss_count"])
    load_count = int(policy["load_count"])
    cold_bytes = miss_count * slot_bytes
    return {
        "policy": policy["policy"],
        "demand_count": demand_count,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "load_count": load_count,
        "hit_rate": hit_count / demand_count,
        "cold_bytes": cold_bytes,
        "cold_bytes_per_token": cold_bytes / token_count,
        "serialized_transfer_ms": miss_count * expert_transfer_ms,
        "serialized_transfer_ms_per_token": (
            miss_count * expert_transfer_ms / token_count
        ),
    }


def _combine_policy_metrics(
    rows: list[dict[str, object]],
    *,
    token_count: int,
    slot_bytes: int,
    expert_transfer_ms: float,
) -> dict[str, object]:
    if not rows:
        raise ValueError("policy metric rows must not be empty")
    demand_count = sum(int(row["demand_count"]) for row in rows)
    hit_count = sum(int(row["hit_count"]) for row in rows)
    miss_count = sum(int(row["miss_count"]) for row in rows)
    load_count = sum(int(row["load_count"]) for row in rows)
    cold_bytes = miss_count * slot_bytes
    return {
        "policy": rows[0]["policy"],
        "demand_count": demand_count,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "load_count": load_count,
        "hit_rate": hit_count / demand_count,
        "cold_bytes": cold_bytes,
        "cold_bytes_per_token": cold_bytes / token_count,
        "serialized_transfer_ms": miss_count * expert_transfer_ms,
        "serialized_transfer_ms_per_token": (
            miss_count * expert_transfer_ms / token_count
        ),
    }


def build_heldout_breakdown(
    training_events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    evaluation_traces: list[EvaluationTrace] | tuple[EvaluationTrace, ...],
    *,
    capacity_per_layer: int,
    slot_bytes: int,
    expert_transfer_ms: float,
) -> dict[str, object]:
    """Fit static residents once and reset online LRU per conversation."""

    training = tuple(training_events)
    evaluations = tuple(evaluation_traces)
    if not training or not evaluations or any(not item.events for item in evaluations):
        raise ValueError("training and evaluation traces must not be empty")
    if capacity_per_layer <= 0 or slot_bytes <= 0 or expert_transfer_ms <= 0:
        raise ValueError("capacity, slot bytes, and transfer time must be positive")
    identifiers = [item.conversation_id for item in evaluations]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("evaluation conversation IDs must be unique")
    training_layers = {event.layer_id for event in training}
    if any(
        {event.layer_id for event in item.events} != training_layers
        for item in evaluations
    ):
        raise ValueError("training and evaluation layer sets must match")

    per_prompt: list[dict[str, object]] = []
    for item in evaluations:
        token_count = _token_count(item.events)
        curve = build_held_out_capacity_curve(
            training,
            item.events,
            capacities=(capacity_per_layer,),
            slot_bytes=slot_bytes,
            expert_transfer_ms=expert_transfer_ms,
        )
        point = curve["points"][0]
        per_prompt.append(
            {
                "conversation_id": item.conversation_id,
                "split": item.split,
                "domain": item.domain,
                "source_trace": item.source_trace,
                "event_count": len(item.events),
                "token_count": token_count,
                "expert_demand_count": sum(
                    len(event.selected_expert_ids) for event in item.events
                ),
                "static_hotset": _policy_metrics(
                    point["static_hotset"],
                    token_count=token_count,
                    slot_bytes=slot_bytes,
                    expert_transfer_ms=expert_transfer_ms,
                ),
                "lru": _policy_metrics(
                    point["lru"],
                    token_count=token_count,
                    slot_bytes=slot_bytes,
                    expert_transfer_ms=expert_transfer_ms,
                ),
            }
        )

    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_prompt:
        grouped[str(row["domain"])].append(row)

    def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
        token_count = sum(int(row["token_count"]) for row in rows)
        return {
            "conversation_count": len(rows),
            "event_count": sum(int(row["event_count"]) for row in rows),
            "token_count": token_count,
            "expert_demand_count": sum(
                int(row["expert_demand_count"]) for row in rows
            ),
            "static_hotset": _combine_policy_metrics(
                [row["static_hotset"] for row in rows],
                token_count=token_count,
                slot_bytes=slot_bytes,
                expert_transfer_ms=expert_transfer_ms,
            ),
            "lru": _combine_policy_metrics(
                [row["lru"] for row in rows],
                token_count=token_count,
                slot_bytes=slot_bytes,
                expert_transfer_ms=expert_transfer_ms,
            ),
        }

    per_domain = [
        {"domain": domain, **aggregate_rows(rows)}
        for domain, rows in sorted(grouped.items())
    ]
    aggregate = aggregate_rows(per_prompt)
    return {
        "schema_version": "1.0.0",
        "measurement_kind": "estimated_policy_over_measured_routing",
        "fit_scope": "held_out_conversation_split",
        "lru_reset_scope": "conversation",
        "capacity_per_layer": capacity_per_layer,
        "slot_bytes": slot_bytes,
        "expert_transfer_ms": expert_transfer_ms,
        "layer_ids": sorted(training_layers),
        "training_event_count": len(training),
        "evaluation_event_count": sum(
            len(item.events) for item in evaluations
        ),
        "per_prompt": per_prompt,
        "per_domain": per_domain,
        "aggregate": aggregate,
    }
