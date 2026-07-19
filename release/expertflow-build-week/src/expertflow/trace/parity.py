"""Strict token-sequence artifacts and measured parity reports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from expertflow.trace.schema import SCHEMA_VERSION


class TokenSequenceError(ValueError):
    """Raised when a token artifact is malformed or ambiguous."""


@dataclass(frozen=True, slots=True)
class TokenSequence:
    """Prompt and generated token IDs emitted by a deterministic probe run."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]


_FIELDS = frozenset(
    {"schema_version", "prompt_token_ids", "generated_token_ids"}
)


def _token_ids(record: dict[str, object], field: str) -> tuple[int, ...]:
    values = record.get(field)
    if not isinstance(values, list):
        raise TokenSequenceError(f"{field} must be an array")

    tokens: list[int] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 4_294_967_295
        ):
            raise TokenSequenceError(
                f"{field} must contain unsigned 32-bit integers"
            )
        tokens.append(value)
    return tuple(tokens)


def load_token_sequence(path: Path) -> TokenSequence:
    """Load a strict, versioned token-sequence JSON artifact."""

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TokenSequenceError(f"cannot read token sequence: {error}") from error

    if not isinstance(record, dict):
        raise TokenSequenceError("token sequence must be a JSON object")

    unknown = set(record) - _FIELDS
    missing = _FIELDS - set(record)
    if unknown:
        raise TokenSequenceError(f"unknown fields: {sorted(unknown)}")
    if missing:
        raise TokenSequenceError(f"missing fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise TokenSequenceError(
            "unsupported schema_version "
            f"{record['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )

    return TokenSequence(
        prompt_token_ids=_token_ids(record, "prompt_token_ids"),
        generated_token_ids=_token_ids(record, "generated_token_ids"),
    )


def _first_mismatch(
    baseline: tuple[int, ...], instrumented: tuple[int, ...]
) -> dict[str, int | None] | None:
    for index in range(max(len(baseline), len(instrumented))):
        baseline_token = baseline[index] if index < len(baseline) else None
        instrumented_token = (
            instrumented[index] if index < len(instrumented) else None
        )
        if baseline_token != instrumented_token:
            return {
                "index": index,
                "baseline_token_id": baseline_token,
                "instrumented_token_id": instrumented_token,
            }
    return None


def compare_token_sequences(
    baseline_path: Path, instrumented_path: Path
) -> dict[str, object]:
    """Compare two real probe artifacts without weakening exact equality."""

    baseline = load_token_sequence(baseline_path)
    instrumented = load_token_sequence(instrumented_path)
    first_generated_mismatch = _first_mismatch(
        baseline.generated_token_ids, instrumented.generated_token_ids
    )

    return {
        "classification": "measured",
        "baseline_path": str(baseline_path.resolve()),
        "instrumented_path": str(instrumented_path.resolve()),
        "prompt_matches": (
            baseline.prompt_token_ids == instrumented.prompt_token_ids
        ),
        "generated_matches": first_generated_mismatch is None,
        "baseline_generated_count": len(baseline.generated_token_ids),
        "instrumented_generated_count": len(
            instrumented.generated_token_ids
        ),
        "first_generated_mismatch": first_generated_mismatch,
    }
