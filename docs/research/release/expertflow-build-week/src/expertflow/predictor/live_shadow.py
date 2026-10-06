"""Strict validation for P1 live-shadow predictor records."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from expertflow.predictor.runtime_artifact import (
    RuntimeArtifact,
    predict_runtime_artifact,
)


@dataclass(frozen=True, slots=True)
class ShadowRecord:
    run_id: str
    forward_index: int
    phase: str
    phase_generation: int
    source_experts: tuple[int, ...]
    predicted_experts: tuple[int, ...]
    predicted_scores: tuple[float, ...]
    actual_experts: tuple[int, ...]
    recall_at_8_matches: int
    recall_at_12_matches: int
    prediction_latency_ns: int
    artifact_sha256: str
    configuration_sha256: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
        values.append(value)
    return values


def load_shadow_log(
    path: Path,
) -> tuple[tuple[ShadowRecord, ...], dict[str, Any]]:
    values = _read_jsonl(path)
    if not values or values[-1].get("record_kind") != "summary":
        raise ValueError("shadow log has no final summary")
    summary = values[-1]
    if summary.get("pending_transition") is not False:
        raise ValueError("shadow log has a pending transition")
    if summary.get("candidate_support_failures") != 0:
        raise ValueError("shadow log contains candidate-support failures")

    records: list[ShadowRecord] = []
    seen: set[tuple[str, int, str]] = set()
    for value in values[:-1]:
        if value.get("schema_version") != "1.0.0":
            raise ValueError("shadow transition schema is unsupported")
        if value.get("record_kind") != "transition":
            raise ValueError("shadow log contains an unexpected record kind")
        if value.get("source_layer") != 23 or value.get("target_layer") != 24:
            raise ValueError("shadow transition layer pair is invalid")
        record = ShadowRecord(
            run_id=str(value["run_id"]),
            forward_index=int(value["forward_index"]),
            phase=str(value["phase"]),
            phase_generation=int(value["phase_generation"]),
            source_experts=tuple(int(item) for item in value["source_experts"]),
            predicted_experts=tuple(
                int(item) for item in value["predicted_experts"]
            ),
            predicted_scores=tuple(
                float(item) for item in value["predicted_scores"]
            ),
            actual_experts=tuple(int(item) for item in value["actual_experts"]),
            recall_at_8_matches=int(value["recall_at_8_matches"]),
            recall_at_12_matches=int(value["recall_at_12_matches"]),
            prediction_latency_ns=int(value["prediction_latency_ns"]),
            artifact_sha256=str(value["artifact_sha256"]),
            configuration_sha256=str(value["configuration_sha256"]),
        )
        if (
            record.phase not in {"prefill", "decode"}
            or record.phase_generation <= 0
            or len(record.source_experts) != 8
            or len(record.predicted_experts) != 12
            or len(record.predicted_scores) != 12
            or len(record.actual_experts) != 8
            or len(set(record.source_experts)) != 8
            or len(set(record.predicted_experts)) != 12
            or len(set(record.actual_experts)) != 8
            or record.prediction_latency_ns < 0
            or any(not math.isfinite(score) for score in record.predicted_scores)
        ):
            raise ValueError("shadow transition contract is invalid")
        identity = (record.run_id, record.forward_index, record.phase)
        if identity in seen:
            raise ValueError("shadow log contains duplicate transitions")
        seen.add(identity)
        records.append(record)
    if int(summary.get("transitions", -1)) != len(records):
        raise ValueError("shadow summary transition count does not match")
    if records and any(record.run_id != summary.get("run_id") for record in records):
        raise ValueError("shadow run identifier is inconsistent")
    return tuple(records), summary


def validate_offline_equivalence(
    artifact: RuntimeArtifact,
    records: tuple[ShadowRecord, ...],
) -> None:
    for record in records:
        expected_ids, expected_scores = predict_runtime_artifact(
            artifact,
            phase=record.phase,
            source_expert_ids=record.source_experts,
        )
        if record.predicted_experts != expected_ids:
            raise ValueError(
                f"offline/live candidate mismatch at forward {record.forward_index}"
            )
        if record.predicted_scores != expected_scores:
            raise ValueError(
                f"offline/live score mismatch at forward {record.forward_index}"
            )
        if record.artifact_sha256 != artifact.payload_sha256:
            raise ValueError("shadow artifact hash does not match")
        if (
            record.configuration_sha256
            != artifact.identity.configuration_sha256
        ):
            raise ValueError("shadow predictor configuration hash does not match")
        matches_8 = sum(
            expert in record.predicted_experts[:8]
            for expert in record.actual_experts
        )
        matches_12 = sum(
            expert in record.predicted_experts
            for expert in record.actual_experts
        )
        if (
            record.recall_at_8_matches != matches_8
            or record.recall_at_12_matches != matches_12
        ):
            raise ValueError("shadow recall contribution is invalid")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_shadow_records(
    records: tuple[ShadowRecord, ...],
    summary: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "transitions": len(records),
        "candidate_support_failures": int(
            summary["candidate_support_failures"]
        ),
    }
    for phase in ("prefill", "decode"):
        phase_records = [record for record in records if record.phase == phase]
        latencies_us = [
            record.prediction_latency_ns / 1000.0 for record in phase_records
        ]
        denominator = 8 * len(phase_records)
        result[phase] = {
            "transitions": len(phase_records),
            "recall_at_8": (
                sum(record.recall_at_8_matches for record in phase_records)
                / denominator
                if denominator
                else 0.0
            ),
            "recall_at_12": (
                sum(record.recall_at_12_matches for record in phase_records)
                / denominator
                if denominator
                else 0.0
            ),
            "latency_p50_us": _percentile(latencies_us, 0.50),
            "latency_p95_us": _percentile(latencies_us, 0.95),
        }
    return result


def _router_projection(path: Path) -> tuple[tuple[Any, ...], ...]:
    projection = []
    for value in _read_jsonl(path):
        projection.append(
            (
                value.get("phase"),
                value.get("forward_id"),
                value.get("token_index"),
                value.get("token_id"),
                value.get("layer_id"),
                tuple(value.get("selected_expert_ids", [])),
            )
        )
    return tuple(projection)


def validate_token_and_router_parity(
    disabled_tokens: Path,
    enabled_tokens: Path,
    disabled_trace: Path,
    enabled_trace: Path,
) -> None:
    if json.loads(disabled_tokens.read_text(encoding="utf-8")) != json.loads(
        enabled_tokens.read_text(encoding="utf-8")
    ):
        raise ValueError("prompt or generated token parity failed")
    if _router_projection(disabled_trace) != _router_projection(enabled_trace):
        raise ValueError("router parity failed")
