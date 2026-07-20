"""Versioned runtime artifact for the frozen B2 transition predictor."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from typing import Iterable

from expertflow.predictor.models import TransitionPredictor


MAGIC = b"EFB2RT01"
FORMAT_VERSION = 1
EXPERT_COUNT = 128
SOURCE_WIDTH = 8
CANDIDATE_WIDTH = 12
SOURCE_LAYER = 23
TARGET_LAYER = 24
PHASES = ("prefill", "decode")
SCORING = b"source_normalized"
TABLES = b"phase_separated"
ADMISSION = b"observed_support"
TIE_BREAK = b"ascending_expert_id"
HEADER = struct.Struct("<8sII32s32s6H24s24s24s24s32s32sQ32s")
SCORE_COUNT = len(PHASES) * EXPERT_COUNT * EXPERT_COUNT
SUPPORT_BYTES = len(PHASES) * EXPERT_COUNT * (EXPERT_COUNT // 8)
FALLBACK_COUNT = EXPERT_COUNT
PAYLOAD_BYTES = (SCORE_COUNT + FALLBACK_COUNT) * 8 + SUPPORT_BYTES


def _fixed(value: bytes, width: int) -> bytes:
    if len(value) > width:
        raise ValueError("fixed artifact identifier is too long")
    return value.ljust(width, b"\0")


def _hex_digest(value: str, name: str) -> bytes:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest") from exc


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    model_sha256: str
    runtime_sha256: str
    manifest_sha256: str
    configuration_sha256: str


def artifact_identity_payload(identity: ArtifactIdentity) -> dict[str, str]:
    return {
        "model_sha256": identity.model_sha256,
        "runtime_sha256": identity.runtime_sha256,
        "manifest_sha256": identity.manifest_sha256,
        "configuration_sha256": identity.configuration_sha256,
    }


@dataclass(frozen=True, slots=True)
class RuntimeArtifact:
    format_version: int
    source_layer: int
    target_layer: int
    expert_count: int
    source_width: int
    candidate_width: int
    identity: ArtifactIdentity
    payload_sha256: str
    payload: bytes
    scores: dict[str, tuple[tuple[float, ...], ...]]
    support: dict[str, tuple[int, ...]]
    fallback_scores: tuple[float, ...]


def _table_payload(
    predictor: TransitionPredictor,
) -> tuple[bytes, dict[str, tuple[tuple[float, ...], ...]], dict[str, tuple[int, ...]]]:
    score_values: list[float] = []
    tables: dict[str, tuple[tuple[float, ...], ...]] = {}
    support: dict[str, tuple[int, ...]] = {}
    support_payload = bytearray()
    for phase in PHASES:
        phase_rows: list[tuple[float, ...]] = []
        phase_masks: list[int] = []
        by_source = predictor.transitions.get(phase, {}).get(TARGET_LAYER, {})
        for source_expert in range(EXPERT_COUNT):
            counts = by_source.get(source_expert)
            total = counts.total() if counts is not None else 0
            row = tuple(
                (float(counts[target]) / total if total else 0.0)
                for target in range(EXPERT_COUNT)
            )
            phase_rows.append(row)
            score_values.extend(row)
            mask = sum(
                1 << target
                for target, score in enumerate(row)
                if score > 0.0
            )
            phase_masks.append(mask)
            support_payload.extend(mask.to_bytes(EXPERT_COUNT // 8, "little"))
        tables[phase] = tuple(phase_rows)
        support[phase] = tuple(phase_masks)
    score_payload = struct.pack(f"<{len(score_values)}d", *score_values)
    return score_payload + support_payload, tables, support


def build_runtime_artifact(
    predictor: TransitionPredictor,
    identity: ArtifactIdentity,
) -> bytes:
    if predictor.weighting != "source_normalized":
        raise ValueError("runtime artifact requires source-normalized scoring")
    if predictor.phase_mode != "separate":
        raise ValueError("runtime artifact requires phase-separated tables")
    table_payload, _, _ = _table_payload(predictor)
    fallback = predictor.fallback.counts.get(TARGET_LAYER)
    fallback_scores = tuple(
        float(fallback[expert]) if fallback is not None else 0.0
        for expert in range(EXPERT_COUNT)
    )
    payload = table_payload + struct.pack(f"<{EXPERT_COUNT}d", *fallback_scores)
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError("runtime artifact payload dimensions are invalid")
    payload_digest = sha256(payload).digest()
    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        HEADER.size,
        _hex_digest(identity.model_sha256, "model_sha256"),
        _hex_digest(identity.runtime_sha256, "runtime_sha256"),
        SOURCE_LAYER,
        TARGET_LAYER,
        EXPERT_COUNT,
        SOURCE_WIDTH,
        CANDIDATE_WIDTH,
        0,
        _fixed(SCORING, 24),
        _fixed(TABLES, 24),
        _fixed(ADMISSION, 24),
        _fixed(TIE_BREAK, 24),
        _hex_digest(identity.manifest_sha256, "manifest_sha256"),
        _hex_digest(identity.configuration_sha256, "configuration_sha256"),
        len(payload),
        payload_digest,
    )
    return header + payload


def _decode_identifier(value: bytes) -> str:
    return value.rstrip(b"\0").decode("ascii")


def parse_runtime_artifact(data: bytes) -> RuntimeArtifact:
    if len(data) < HEADER.size:
        raise ValueError("runtime artifact is truncated")
    fields = HEADER.unpack_from(data)
    (
        magic,
        version,
        header_size,
        model_digest,
        runtime_digest,
        source_layer,
        target_layer,
        expert_count,
        source_width,
        candidate_width,
        reserved,
        scoring,
        tables,
        admission,
        tie_break,
        manifest_digest,
        configuration_digest,
        payload_bytes,
        payload_digest,
    ) = fields
    if magic != MAGIC:
        raise ValueError("runtime artifact magic is invalid")
    if version != FORMAT_VERSION:
        raise ValueError("runtime artifact version is unsupported")
    if header_size != HEADER.size or reserved != 0:
        raise ValueError("runtime artifact header dimensions are invalid")
    if (
        source_layer != SOURCE_LAYER
        or target_layer != TARGET_LAYER
        or expert_count != EXPERT_COUNT
        or source_width != SOURCE_WIDTH
        or candidate_width != CANDIDATE_WIDTH
    ):
        raise ValueError("runtime artifact predictor dimensions are invalid")
    if (
        _decode_identifier(scoring) != SCORING.decode()
        or _decode_identifier(tables) != TABLES.decode()
        or _decode_identifier(admission) != ADMISSION.decode()
        or _decode_identifier(tie_break) != TIE_BREAK.decode()
    ):
        raise ValueError("runtime artifact predictor configuration is invalid")
    if payload_bytes != PAYLOAD_BYTES:
        raise ValueError("runtime artifact payload dimensions are invalid")
    if len(data) < HEADER.size + payload_bytes:
        raise ValueError("runtime artifact is truncated")
    if len(data) != HEADER.size + payload_bytes:
        raise ValueError("runtime artifact has trailing bytes")
    payload = data[HEADER.size :]
    if sha256(payload).digest() != payload_digest:
        raise ValueError("runtime artifact payload checksum does not match")

    offset = 0
    tables_by_phase: dict[str, tuple[tuple[float, ...], ...]] = {}
    for phase in PHASES:
        rows: list[tuple[float, ...]] = []
        for _ in range(EXPERT_COUNT):
            row = struct.unpack_from(f"<{EXPERT_COUNT}d", payload, offset)
            offset += EXPERT_COUNT * 8
            if any(not math.isfinite(value) or value < 0.0 for value in row):
                raise ValueError("runtime artifact contains invalid scores")
            rows.append(tuple(row))
        tables_by_phase[phase] = tuple(rows)

    support_by_phase: dict[str, tuple[int, ...]] = {}
    for phase in PHASES:
        masks: list[int] = []
        for _ in range(EXPERT_COUNT):
            mask = int.from_bytes(
                payload[offset : offset + EXPERT_COUNT // 8], "little"
            )
            offset += EXPERT_COUNT // 8
            masks.append(mask)
        support_by_phase[phase] = tuple(masks)

    fallback = struct.unpack_from(f"<{EXPERT_COUNT}d", payload, offset)
    if any(not math.isfinite(value) or value < 0.0 for value in fallback):
        raise ValueError("runtime artifact contains invalid fallback scores")
    offset += EXPERT_COUNT * 8
    if offset != len(payload):
        raise ValueError("runtime artifact payload dimensions are invalid")

    for phase in PHASES:
        for source, row in enumerate(tables_by_phase[phase]):
            expected = sum(
                1 << target
                for target, score in enumerate(row)
                if score > 0.0
            )
            if expected != support_by_phase[phase][source]:
                raise ValueError("runtime artifact observed-support mask is invalid")

    return RuntimeArtifact(
        format_version=version,
        source_layer=source_layer,
        target_layer=target_layer,
        expert_count=expert_count,
        source_width=source_width,
        candidate_width=candidate_width,
        identity=ArtifactIdentity(
            model_sha256=model_digest.hex(),
            runtime_sha256=runtime_digest.hex(),
            manifest_sha256=manifest_digest.hex(),
            configuration_sha256=configuration_digest.hex(),
        ),
        payload_sha256=payload_digest.hex(),
        payload=payload,
        scores=tables_by_phase,
        support=support_by_phase,
        fallback_scores=tuple(fallback),
    )


def predict_runtime_artifact(
    artifact: RuntimeArtifact,
    *,
    phase: str,
    source_expert_ids: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if phase not in PHASES:
        raise ValueError("predictor phase must be prefill or decode")
    if (
        len(source_expert_ids) != artifact.source_width
        or len(set(source_expert_ids)) != artifact.source_width
        or any(not 0 <= expert < artifact.expert_count for expert in source_expert_ids)
    ):
        raise ValueError("source expert IDs must be eight unique valid IDs")
    scores = [
        sum(artifact.scores[phase][source][target] for source in source_expert_ids)
        for target in range(artifact.expert_count)
    ]
    supported = [
        target
        for target, score in enumerate(scores)
        if score > 0.0
    ]
    if len(supported) < artifact.candidate_width:
        raise ValueError("fewer than twelve candidates have observed support")
    ranked = sorted(supported, key=lambda expert: (-scores[expert], expert))
    selected = tuple(ranked[: artifact.candidate_width])
    return selected, tuple(scores[expert] for expert in selected)


def write_parity_fixtures(
    artifact: RuntimeArtifact,
    cases: Iterable[tuple[str, tuple[int, ...]]],
    output: Path,
) -> None:
    fixtures = []
    for phase, source in cases:
        candidates, scores = predict_runtime_artifact(
            artifact, phase=phase, source_expert_ids=source
        )
        fixtures.append(
            {
                "phase": phase,
                "source_expert_ids": list(source),
                "predicted_expert_ids": list(candidates),
                "predicted_scores": list(scores),
            }
        )
    payload = {
        "schema_version": "1.0.0",
        "artifact_format_version": artifact.format_version,
        "artifact_payload_sha256": artifact.payload_sha256,
        "candidate_width": artifact.candidate_width,
        "fixtures": fixtures,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
