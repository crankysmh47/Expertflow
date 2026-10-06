"""Strict validation and analysis for T1 temporal live-shadow telemetry."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import fmean, median

from expertflow.predictor.temporal_runtime_artifact import (
    TemporalRuntimeArtifact,
    predict_temporal_runtime_artifact,
)


H2D_NS = 306_016
STAGING_NS = 1_025_400
QUEUE_NS = 22_200
SAFETY_NS = 250_000


@dataclass(frozen=True, slots=True)
class TemporalShadowRecord:
    run_id: str
    conversation_generation: int
    source_forward_index: int
    target_forward_index: int
    source_decode_index: int
    target_decode_index: int
    source_observed_ns: int
    predictor_finished_ns: int
    target_observed_ns: int
    prediction_latency_ns: int
    source_experts: tuple[int, ...]
    session_counts_after_source: tuple[int, ...]
    predicted_experts: tuple[int, ...]
    predicted_scores: tuple[float, ...]
    actual_experts: tuple[int, ...]
    artifact_sha256: str
    configuration_sha256: str


def _ids(value: object, width: int, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(isinstance(expert, bool) or not isinstance(expert, int) or not 0 <= expert < 128 for expert in value)
        or len(set(value)) != width
    ):
        raise ValueError(f"{name} must contain {width} unique expert IDs")
    return tuple(value)


def load_temporal_shadow_log(
    path: Path,
) -> tuple[tuple[TemporalShadowRecord, ...], dict[str, object]]:
    records = []
    summary = None
    run_id = None
    generation = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("schema_version") != "1.0.0":
            raise ValueError(f"temporal log row {line_number} is invalid")
        kind = row.get("record_kind")
        if kind == "summary":
            if summary is not None or line_number == 1 and records:
                raise ValueError("temporal log has duplicate summary")
            summary = row
            continue
        if kind != "transition" or summary is not None:
            raise ValueError("temporal transition ordering is invalid")
        current_run = row.get("run_id")
        current_generation = row.get("conversation_generation")
        if not isinstance(current_run, str) or not current_run:
            raise ValueError("temporal run ID is invalid")
        if not isinstance(current_generation, int) or current_generation <= 0:
            raise ValueError("temporal conversation generation is invalid")
        if run_id is None:
            run_id, generation = current_run, current_generation
        if current_run != run_id or current_generation != generation:
            raise ValueError("temporal state leaked across run or conversation generation")
        source_forward = row.get("source_forward_index")
        target_forward = row.get("target_forward_index")
        source_decode = row.get("source_decode_index")
        target_decode = row.get("target_decode_index")
        if (
            not all(isinstance(value, int) and value >= 0 for value in (
                source_forward, target_forward, source_decode, target_decode
            ))
            or target_forward != source_forward + 1
            or target_decode != source_decode + 1
        ):
            raise ValueError("temporal identities are not consecutive")
        timestamps = (
            row.get("source_observed_ns"),
            row.get("predictor_finished_ns"),
            row.get("target_observed_ns"),
        )
        if (
            not all(isinstance(value, int) and value >= 0 for value in timestamps)
            or not timestamps[0] <= timestamps[1] <= timestamps[2]
            or row.get("prediction_latency_ns") != timestamps[1] - timestamps[0]
        ):
            raise ValueError("temporal timestamps are invalid")
        counts = row.get("session_counts_after_source")
        if (
            not isinstance(counts, list)
            or len(counts) != 128
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts)
        ):
            raise ValueError("temporal session counts are invalid")
        scores = row.get("predicted_scores")
        if (
            not isinstance(scores, list)
            or len(scores) != 16
            or any(not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)) for score in scores)
        ):
            raise ValueError("temporal predicted scores are invalid")
        records.append(TemporalShadowRecord(
            run_id=current_run,
            conversation_generation=current_generation,
            source_forward_index=source_forward,
            target_forward_index=target_forward,
            source_decode_index=source_decode,
            target_decode_index=target_decode,
            source_observed_ns=timestamps[0],
            predictor_finished_ns=timestamps[1],
            target_observed_ns=timestamps[2],
            prediction_latency_ns=row["prediction_latency_ns"],
            source_experts=_ids(row.get("source_experts"), 8, "source experts"),
            session_counts_after_source=tuple(counts),
            predicted_experts=_ids(row.get("predicted_experts"), 16, "predicted experts"),
            predicted_scores=tuple(float(score) for score in scores),
            actual_experts=_ids(row.get("actual_experts"), 8, "actual experts"),
            artifact_sha256=str(row.get("artifact_sha256")),
            configuration_sha256=str(row.get("configuration_sha256")),
        ))
    if summary is None or not records:
        raise ValueError("temporal log is incomplete")
    if summary.get("transitions") != len(records):
        raise ValueError("temporal summary transition count does not match")
    if summary.get("pending_prediction") is not True:
        raise ValueError("temporal teardown must retain exactly the final pending prediction")
    if summary.get("run_id") != run_id or summary.get("conversation_generation") != generation:
        raise ValueError("temporal summary identity does not match")
    return tuple(records), summary


def validate_temporal_offline_equivalence(
    artifact: TemporalRuntimeArtifact,
    records: tuple[TemporalShadowRecord, ...],
) -> None:
    for record in records:
        if (
            record.artifact_sha256 != artifact.payload_sha256
            or record.configuration_sha256 != artifact.identity.configuration_sha256
        ):
            raise ValueError("temporal artifact identity does not match")
        before = Counter({
            expert: count
            for expert, count in enumerate(record.session_counts_after_source)
            if count
        })
        for expert in record.source_experts:
            before[expert] -= 1
            if before[expert] < 0:
                raise ValueError("temporal session state cannot precede the source update")
            if before[expert] == 0:
                del before[expert]
        candidates, scores, after = predict_temporal_runtime_artifact(
            artifact,
            source_expert_ids=record.source_experts,
            session_counts=before,
        )
        if candidates != record.predicted_experts or scores != record.predicted_scores:
            raise ValueError("live temporal prediction differs from offline prediction")
        if tuple(after[index] for index in range(128)) != record.session_counts_after_source:
            raise ValueError("live temporal session state differs from offline state")


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def nearest(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]
    return {
        "minimum": ordered[0],
        "p5": nearest(0.05),
        "p50": float(median(ordered)),
        "p95": nearest(0.95),
        "maximum": ordered[-1],
        "mean": fmean(ordered),
    }


def _demand(cache: OrderedDict[int, None], expert: int, capacity: int = 32) -> None:
    if expert in cache:
        cache.move_to_end(expert)
        return
    if len(cache) >= capacity:
        cache.popitem(last=False)
    cache[expert] = None


def summarize_temporal_shadow(
    records: tuple[TemporalShadowRecord, ...],
    runtime_summary: dict[str, object],
) -> dict[str, object]:
    if not records or runtime_summary.get("transitions") != len(records):
        raise ValueError("temporal records are incomplete")
    predictor_us = [record.prediction_latency_ns / 1000.0 for record in records]
    lead_ns = [record.target_observed_ns - record.predictor_finished_ns for record in records]
    lead_us = [value / 1000.0 for value in lead_ns]
    ranking = Counter()
    highest_ranks = []
    reciprocal_ranks = []
    cache: OrderedDict[int, None] = OrderedDict()
    for expert in records[0].source_experts:
        _demand(cache, expert)
    one_transfer = Counter()
    one_transfer_events = []
    full_deadline = STAGING_NS + QUEUE_NS + H2D_NS + SAFETY_NS
    for record, available_ns in zip(records, lead_ns, strict=True):
        actual = set(record.actual_experts)
        positions = [
            index + 1 for index, expert in enumerate(record.predicted_experts)
            if expert in actual
        ]
        best_rank = min(positions) if positions else 0
        highest_ranks.append(best_rank)
        reciprocal_ranks.append(1.0 / best_rank if best_rank else 0.0)
        for width in (1, 2, 4):
            ranking[f"hit_at_{width}"] += any(
                expert in actual for expert in record.predicted_experts[:width]
            )
        for width in (8, 12, 16):
            ranking[f"recall_at_{width}_matches"] += len(
                actual & set(record.predicted_experts[:width])
            )
        resident_candidates = sum(expert in cache for expert in record.predicted_experts)
        ranking["resident_candidates"] += resident_candidates
        chosen = next(
            (
                (rank, expert)
                for rank, expert in enumerate(record.predicted_experts, 1)
                if expert not in cache
            ),
            None,
        )
        if chosen is None:
            one_transfer["no_admission"] += 1
            one_transfer_events.append({
                "source_decode_index": record.source_decode_index,
                "target_decode_index": record.target_decode_index,
                "expert_id": None,
                "rank": None,
                "useful": False,
                "ready": False,
                "victim_expert_id": None,
                "eviction_regret": False,
            })
        else:
            rank, expert = chosen
            one_transfer["admitted"] += 1
            one_transfer["chosen_rank_total"] += rank
            victim = next(iter(cache)) if len(cache) >= 32 else None
            if victim is not None and victim in actual:
                one_transfer["eviction_regret"] += 1
            useful = expert in actual
            if useful:
                one_transfer["useful"] += 1
                if available_ns >= full_deadline:
                    one_transfer["estimated_ready_useful"] += 1
                    one_transfer["potential_blocking_misses_avoided"] += 1
                else:
                    one_transfer["estimated_late_useful"] += 1
            else:
                one_transfer["wasted"] += 1
            one_transfer_events.append({
                "source_decode_index": record.source_decode_index,
                "target_decode_index": record.target_decode_index,
                "expert_id": expert,
                "rank": rank,
                "useful": useful,
                "ready": useful and available_ns >= full_deadline,
                "victim_expert_id": victim,
                "eviction_regret": victim is not None and victim in actual,
            })
        for expert in record.actual_experts:
            _demand(cache, expert)
    demands = len(records) * 8
    return {
        "transitions": len(records),
        "prediction_latency_us": _percentiles(predictor_us),
        "lead_time_us": _percentiles(lead_us),
        "deadline_reference": {
            "h2d_cuda_event_us": H2D_NS / 1000.0,
            "staging_us": STAGING_NS / 1000.0,
            "queue_us": QUEUE_NS / 1000.0,
            "safety_margin_us": SAFETY_NS / 1000.0,
        },
        "deadline_eligible": {
            "h2d_only": sum(value >= H2D_NS for value in lead_ns),
            "staging_plus_h2d": sum(value >= STAGING_NS + H2D_NS for value in lead_ns),
            "full_conservative": sum(value >= full_deadline for value in lead_ns),
        },
        "ranking": {
            "hit_at_1": ranking["hit_at_1"],
            "hit_at_2": ranking["hit_at_2"],
            "hit_at_4": ranking["hit_at_4"],
            "hit_at_1_rate": ranking["hit_at_1"] / len(records),
            "hit_at_2_rate": ranking["hit_at_2"] / len(records),
            "hit_at_4_rate": ranking["hit_at_4"] / len(records),
            "recall_at_8": ranking["recall_at_8_matches"] / demands,
            "recall_at_12": ranking["recall_at_12_matches"] / demands,
            "recall_at_16": ranking["recall_at_16_matches"] / demands,
            "highest_true_rank": highest_ranks,
            "mean_reciprocal_rank": fmean(reciprocal_ranks),
            "mean_candidates_already_resident": ranking["resident_candidates"] / len(records),
        },
        "one_transfer_rule": {
            key: one_transfer[key]
            for key in (
                "admitted",
                "useful",
                "wasted",
                "no_admission",
                "estimated_ready_useful",
                "estimated_late_useful",
                "potential_blocking_misses_avoided",
                "eviction_regret",
                "chosen_rank_total",
            )
        } | {
            "useful_rate": one_transfer["useful"] / len(records),
            "wasted_rate": one_transfer["wasted"] / len(records),
            "mean_chosen_rank": (
                one_transfer["chosen_rank_total"] / one_transfer["admitted"]
                if one_transfer["admitted"] else 0.0
            ),
        },
        "one_transfer_events": one_transfer_events,
        "transition_state_memory": {
            key: runtime_summary[key]
            for key in (
                "state_bytes",
                "artifact_bytes",
                "record_bytes",
                "record_capacity",
                "record_storage_bytes",
            )
            if key in runtime_summary
        },
        "measurement_kind": "live_host_wall_shadow_plus_reference_transfer_costs",
        "live_cache_enabled": False,
        "weight_transfers": 0,
    }
