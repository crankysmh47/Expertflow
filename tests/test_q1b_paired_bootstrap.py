from __future__ import annotations

import json
import math

import pytest

from expertflow.quality.q1b import compare_nll_records, load_nll_jsonl


def _records(values: list[float], *, token_offset: int = 0) -> list[dict[str, object]]:
    return [
        {
            "token_index": token_offset + index,
            "chunk_index": index // 4,
            "position": index % 4,
            "token_id": 100 + index,
            "nll": value,
        }
        for index, value in enumerate(values)
    ]


def test_compare_nll_records_reports_exact_point_and_reproducible_interval():
    reference = _records([1.0] * 16)
    candidate = _records([1.005] * 16)

    result = compare_nll_records(
        reference,
        candidate,
        block_size=4,
        bootstrap_samples=250,
        seed=20260718,
    )

    assert result["token_count"] == 16
    assert result["block_count"] == 4
    assert result["relative_perplexity_change"] == pytest.approx(math.expm1(0.005))
    assert result["bootstrap_95pct"]["lower"] == pytest.approx(math.expm1(0.005))
    assert result["bootstrap_95pct"]["upper"] == pytest.approx(math.expm1(0.005))
    assert result["gate_pass"] is True


def test_compare_nll_records_rejects_unpaired_tokens():
    reference = _records([1.0, 1.0])
    candidate = _records([1.0, 1.0], token_offset=1)

    with pytest.raises(ValueError, match="pairing mismatch"):
        compare_nll_records(reference, candidate, block_size=1, bootstrap_samples=10, seed=1)


def test_load_nll_jsonl_rejects_non_finite_values(tmp_path):
    path = tmp_path / "nll.jsonl"
    path.write_text(json.dumps({**_records([1.0])[0], "nll": float("nan")}) + "\n")

    with pytest.raises(ValueError, match="finite"):
        load_nll_jsonl(path)
