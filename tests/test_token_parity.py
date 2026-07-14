import json
from pathlib import Path

import pytest

from expertflow.trace.parity import (
    TokenSequenceError,
    compare_token_sequences,
    load_token_sequence,
)


def write_sequence(
    path: Path,
    *,
    prompt: list[int] | None = None,
    generated: list[int] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "prompt_token_ids": prompt if prompt is not None else [1, 2],
                "generated_token_ids": generated if generated is not None else [3, 4],
            }
        ),
        encoding="utf-8",
    )


def test_reports_exact_measured_token_parity(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    instrumented = tmp_path / "instrumented.json"
    write_sequence(baseline)
    write_sequence(instrumented)

    report = compare_token_sequences(baseline, instrumented)

    assert report["classification"] == "measured"
    assert report["prompt_matches"] is True
    assert report["generated_matches"] is True
    assert report["first_generated_mismatch"] is None


def test_reports_first_generated_token_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    instrumented = tmp_path / "instrumented.json"
    write_sequence(baseline, generated=[3, 4, 5])
    write_sequence(instrumented, generated=[3, 9, 5])

    report = compare_token_sequences(baseline, instrumented)

    assert report["generated_matches"] is False
    assert report["first_generated_mismatch"] == {
        "index": 1,
        "baseline_token_id": 4,
        "instrumented_token_id": 9,
    }


def test_rejects_untrusted_token_sequence(tmp_path: Path) -> None:
    candidate = tmp_path / "tokens.json"
    write_sequence(candidate)
    record = json.loads(candidate.read_text(encoding="utf-8"))
    record["unexpected"] = True
    candidate.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(TokenSequenceError, match="unknown fields"):
        load_token_sequence(candidate)
