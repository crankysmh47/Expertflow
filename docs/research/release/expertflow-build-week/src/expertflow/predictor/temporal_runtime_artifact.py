"""Versioned runtime artifact for the frozen T0 temporal predictor."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import math
import struct

from expertflow.predictor.temporal_models import TemporalTransitionPredictor


MAGIC = b"EFT0RT01"
FORMAT_VERSION = 1
EXPERT_COUNT = 128
SOURCE_WIDTH = 8
CANDIDATE_WIDTH = 16
TARGET_LAYER = 24
WEIGHTS = (0.5, 0.4, 0.1)
HEADER = struct.Struct("<8sII32s32s32s32s4H3dQ32s")
PAYLOAD_BYTES = EXPERT_COUNT * EXPERT_COUNT * 8


@dataclass(frozen=True, slots=True)
class TemporalArtifactIdentity:
    model_sha256: str
    runtime_sha256: str
    manifest_sha256: str
    configuration_sha256: str


@dataclass(frozen=True, slots=True)
class TemporalRuntimeArtifact:
    identity: TemporalArtifactIdentity
    payload_sha256: str
    transition_scores: tuple[tuple[float, ...], ...]


def _digest(value: str, name: str) -> bytes:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest") from error


def _transition_payload(
    predictor: TemporalTransitionPredictor,
) -> tuple[bytes, tuple[tuple[float, ...], ...]]:
    rows = []
    values = []
    for source in range(EXPERT_COUNT):
        counts = predictor.transitions.get(source, Counter())
        total = counts.total()
        row = tuple(
            float(counts[target]) / total if total else 0.0
            for target in range(EXPERT_COUNT)
        )
        rows.append(row)
        values.extend(row)
    return struct.pack(f"<{len(values)}d", *values), tuple(rows)


def build_temporal_runtime_artifact(
    predictor: TemporalTransitionPredictor,
    identity: TemporalArtifactIdentity,
) -> bytes:
    payload, _ = _transition_payload(predictor)
    digest = sha256(payload).digest()
    return HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        HEADER.size,
        _digest(identity.model_sha256, "model_sha256"),
        _digest(identity.runtime_sha256, "runtime_sha256"),
        _digest(identity.manifest_sha256, "manifest_sha256"),
        _digest(identity.configuration_sha256, "configuration_sha256"),
        TARGET_LAYER,
        EXPERT_COUNT,
        SOURCE_WIDTH,
        CANDIDATE_WIDTH,
        *WEIGHTS,
        len(payload),
        digest,
    ) + payload


def parse_temporal_runtime_artifact(data: bytes) -> TemporalRuntimeArtifact:
    if len(data) < HEADER.size:
        raise ValueError("temporal runtime artifact is truncated")
    (
        magic,
        version,
        header_size,
        model_digest,
        runtime_digest,
        manifest_digest,
        configuration_digest,
        layer,
        expert_count,
        source_width,
        candidate_width,
        transition_weight,
        retention_weight,
        session_weight,
        payload_bytes,
        payload_digest,
    ) = HEADER.unpack_from(data)
    if magic != MAGIC or version != FORMAT_VERSION or header_size != HEADER.size:
        raise ValueError("temporal runtime artifact header is invalid")
    if (
        layer != TARGET_LAYER
        or expert_count != EXPERT_COUNT
        or source_width != SOURCE_WIDTH
        or candidate_width != CANDIDATE_WIDTH
        or (transition_weight, retention_weight, session_weight) != WEIGHTS
        or payload_bytes != PAYLOAD_BYTES
    ):
        raise ValueError("temporal runtime artifact configuration is invalid")
    if len(data) != HEADER.size + payload_bytes:
        raise ValueError("temporal runtime artifact dimensions are invalid")
    payload = data[HEADER.size:]
    if sha256(payload).digest() != payload_digest:
        raise ValueError("temporal runtime artifact payload checksum does not match")
    values = struct.unpack(f"<{EXPERT_COUNT * EXPERT_COUNT}d", payload)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("temporal runtime artifact contains non-finite scores")
    rows = tuple(
        tuple(values[source * EXPERT_COUNT:(source + 1) * EXPERT_COUNT])
        for source in range(EXPERT_COUNT)
    )
    return TemporalRuntimeArtifact(
        identity=TemporalArtifactIdentity(
            model_digest.hex(),
            runtime_digest.hex(),
            manifest_digest.hex(),
            configuration_digest.hex(),
        ),
        payload_sha256=payload_digest.hex(),
        transition_scores=rows,
    )


def score_temporal_runtime_artifact(
    artifact: TemporalRuntimeArtifact,
    *,
    source_expert_ids: tuple[int, ...],
    session_counts: Counter[int],
) -> tuple[tuple[float, ...], Counter[int]]:
    if (
        len(source_expert_ids) != SOURCE_WIDTH
        or len(set(source_expert_ids)) != SOURCE_WIDTH
        or any(not 0 <= expert < EXPERT_COUNT for expert in source_expert_ids)
    ):
        raise ValueError("source expert IDs must be eight unique valid IDs")
    updated = Counter(session_counts)
    if any(
        isinstance(expert, bool)
        or not isinstance(expert, int)
        or not 0 <= expert < EXPERT_COUNT
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for expert, count in updated.items()
    ):
        raise ValueError("session counts must contain valid expert IDs and non-negative integers")
    updated.update(source_expert_ids)
    transition = [
        sum(artifact.transition_scores[source][target] for source in source_expert_ids)
        for target in range(EXPERT_COUNT)
    ]
    transition_max = max(transition) or 1.0
    session_max = max(updated.values(), default=0) or 1
    current = set(source_expert_ids)
    scores = tuple(
        WEIGHTS[0] * transition[expert] / transition_max
        + WEIGHTS[1] * (expert in current)
        + WEIGHTS[2] * updated[expert] / session_max
        for expert in range(EXPERT_COUNT)
    )
    if any(not math.isfinite(score) or score < 0.0 for score in scores):
        raise ValueError("temporal predictor produced non-finite scores")
    return scores, updated


def predict_temporal_runtime_artifact(
    artifact: TemporalRuntimeArtifact,
    *,
    source_expert_ids: tuple[int, ...],
    session_counts: Counter[int],
) -> tuple[tuple[int, ...], tuple[float, ...], Counter[int]]:
    scores, updated = score_temporal_runtime_artifact(
        artifact,
        source_expert_ids=source_expert_ids,
        session_counts=session_counts,
    )
    supported = [expert for expert, score in enumerate(scores) if score > 0.0]
    if len(supported) < CANDIDATE_WIDTH:
        raise ValueError("temporal predictor has fewer than sixteen supported candidates")
    supported.sort(key=lambda expert: (-scores[expert], expert))
    candidates = tuple(supported[:CANDIDATE_WIDTH])
    return candidates, tuple(scores[expert] for expert in candidates), updated

