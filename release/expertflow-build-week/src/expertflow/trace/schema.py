"""Strict, versioned schema for real sparse-MoE routing observations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Literal


SCHEMA_VERSION = "1.0.0"
Phase = Literal["prefill", "decode"]


class TraceValidationError(ValueError):
    """Raised when a trace record cannot be trusted."""


@dataclass(frozen=True, slots=True)
class RouterTraceEvent:
    """One token/layer router decision captured from the authoritative graph."""

    schema_version: str
    request_id: str
    conversation_id: str
    turn_index: int
    phase: Phase
    forward_id: int
    hook_order: int
    token_index: int
    token_id: int
    layer_id: int
    selected_expert_ids: tuple[int, ...]
    selected_expert_weights: tuple[float, ...] | None
    observed_at_ns: int


_FIELDS = frozenset(RouterTraceEvent.__dataclass_fields__)


def _fail(record_number: int, reason: str) -> TraceValidationError:
    return TraceValidationError(f"record {record_number}: {reason}")


def _string(record: dict[str, object], field: str, record_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise _fail(record_number, f"{field} must be a non-empty string")
    return value


def _uint(
    record: dict[str, object],
    field: str,
    record_number: int,
    *,
    maximum: int | None = None,
) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail(record_number, f"{field} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise _fail(record_number, f"{field} exceeds {maximum}")
    return value


def _expert_ids(record: dict[str, object], record_number: int) -> tuple[int, ...]:
    values = record.get("selected_expert_ids")
    if not isinstance(values, list) or not values:
        raise _fail(
            record_number, "selected_expert_ids must be a non-empty array"
        )

    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _fail(
                record_number, "selected_expert_ids must contain integers"
            )
        if not 0 <= value <= 65_535:
            raise _fail(
                record_number, "selected_expert_ids must fit unsigned 16-bit"
            )
        result.append(value)

    if len(set(result)) != len(result):
        raise _fail(record_number, "selected_expert_ids must be unique")
    return tuple(result)


def _expert_weights(
    record: dict[str, object],
    expert_count: int,
    record_number: int,
) -> tuple[float, ...] | None:
    values = record.get("selected_expert_weights")
    if values is None:
        return None
    if not isinstance(values, list):
        raise _fail(record_number, "selected_expert_weights must be an array or null")
    if len(values) != expert_count:
        raise _fail(
            record_number,
            "selected_expert_weights must have the same length as selected_expert_ids",
        )

    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _fail(
                record_number, "selected_expert_weights must contain numbers"
            )
        converted = float(value)
        if not math.isfinite(converted) or converted < 0:
            raise _fail(
                record_number,
                "selected_expert_weights must be finite and non-negative",
            )
        result.append(converted)
    return tuple(result)


def parse_router_event(line: str, *, record_number: int) -> RouterTraceEvent:
    """Parse one JSONL record and report failures with its one-based position."""

    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise _fail(record_number, f"invalid JSON: {error.msg}") from error

    if not isinstance(record, dict):
        raise _fail(record_number, "router event must be a JSON object")

    unknown = set(record) - _FIELDS
    missing = _FIELDS - set(record)
    if unknown:
        raise _fail(record_number, f"unknown fields: {sorted(unknown)}")
    if missing:
        raise _fail(record_number, f"missing fields: {sorted(missing)}")

    schema_version = _string(record, "schema_version", record_number)
    if schema_version != SCHEMA_VERSION:
        raise _fail(
            record_number,
            f"unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}",
        )

    phase = _string(record, "phase", record_number)
    if phase not in {"prefill", "decode"}:
        raise _fail(record_number, "phase must be 'prefill' or 'decode'")

    selected_expert_ids = _expert_ids(record, record_number)
    selected_expert_weights = _expert_weights(
        record, len(selected_expert_ids), record_number
    )

    return RouterTraceEvent(
        schema_version=schema_version,
        request_id=_string(record, "request_id", record_number),
        conversation_id=_string(record, "conversation_id", record_number),
        turn_index=_uint(record, "turn_index", record_number, maximum=65_535),
        phase=phase,
        forward_id=_uint(record, "forward_id", record_number),
        hook_order=_uint(record, "hook_order", record_number),
        token_index=_uint(record, "token_index", record_number),
        token_id=_uint(record, "token_id", record_number, maximum=4_294_967_295),
        layer_id=_uint(record, "layer_id", record_number, maximum=65_535),
        selected_expert_ids=selected_expert_ids,
        selected_expert_weights=selected_expert_weights,
        observed_at_ns=_uint(record, "observed_at_ns", record_number),
    )
