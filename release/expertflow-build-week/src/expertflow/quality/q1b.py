from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_PAIR_FIELDS = ("token_index", "chunk_index", "position", "token_id")
_REQUIRED_FIELDS = (*_PAIR_FIELDS, "nll")


def load_nll_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [field for field in _REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(f"line {line_number} missing fields: {missing}")
            if not math.isfinite(float(record["nll"])):
                raise ValueError(f"line {line_number} NLL must be finite")
            records.append(record)
    if not records:
        raise ValueError("NLL file contains no records")
    return records


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate percentile of an empty sequence")
    index = (len(sorted_values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_blocks(
    records: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]], block_size: int
) -> list[list[float]]:
    by_chunk: dict[int, list[float]] = defaultdict(list)
    for reference, candidate in records:
        by_chunk[int(reference["chunk_index"])].append(
            float(candidate["nll"]) - float(reference["nll"])
        )
    return [
        values[start : start + block_size]
        for chunk in sorted(by_chunk)
        for values in (by_chunk[chunk],)
        for start in range(0, len(values), block_size)
    ]


def compare_nll_records(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    block_size: int = 128,
    bootstrap_samples: int = 10_000,
    seed: int = 20260718,
    threshold: float = 0.01,
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("pairing mismatch: record counts differ")
    if block_size <= 0 or bootstrap_samples <= 0:
        raise ValueError("block_size and bootstrap_samples must be positive")

    paired = list(zip(reference, candidate, strict=True))
    for index, (reference_record, candidate_record) in enumerate(paired):
        if any(reference_record[field] != candidate_record[field] for field in _PAIR_FIELDS):
            raise ValueError(f"pairing mismatch at record {index}")
        if not math.isfinite(float(reference_record["nll"])) or not math.isfinite(
            float(candidate_record["nll"])
        ):
            raise ValueError(f"NLL must be finite at record {index}")

    differences = [
        float(candidate_record["nll"]) - float(reference_record["nll"])
        for reference_record, candidate_record in paired
    ]
    blocks = _paired_blocks(paired, block_size)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(bootstrap_samples):
        resampled = [blocks[rng.randrange(len(blocks))] for _ in range(len(blocks))]
        total = sum(sum(block) for block in resampled)
        count = sum(len(block) for block in resampled)
        samples.append(math.expm1(total / count))
    samples.sort()

    point = math.expm1(statistics.fmean(differences))
    lower = _percentile(samples, 0.025)
    upper = _percentile(samples, 0.975)
    return {
        "token_count": len(paired),
        "block_size": block_size,
        "block_count": len(blocks),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "reference_mean_nll": statistics.fmean(float(record["nll"]) for record in reference),
        "candidate_mean_nll": statistics.fmean(float(record["nll"]) for record in candidate),
        "relative_perplexity_change": point,
        "bootstrap_standard_error": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "bootstrap_95pct": {"lower": lower, "upper": upper},
        "threshold": threshold,
        "point_gate_pass": point <= threshold,
        "upper_bound_gate_pass": upper <= threshold,
        "gate_pass": point <= threshold and upper <= threshold,
    }
