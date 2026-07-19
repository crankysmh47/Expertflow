from __future__ import annotations

from typing import Any, Mapping


_REQUIRED = (
    "perplexity",
    "mmlu_correct",
    "mmlu_total",
    "repeated_4gram_rate",
    "distinct_2",
    "deterministic",
    "runtime_clean",
)


def _require_complete(label: str, result: Mapping[str, Any]) -> None:
    missing = [key for key in _REQUIRED if key not in result]
    if missing:
        raise ValueError(f"{label} missing required quality fields: {missing}")
    if result["perplexity"] <= 0:
        raise ValueError(f"{label} perplexity must be positive")
    if result["mmlu_total"] <= 0 or not 0 <= result["mmlu_correct"] <= result["mmlu_total"]:
        raise ValueError(f"{label} MMLU counts are invalid")


def evaluate_quality_gate(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    _require_complete("reference", reference)
    _require_complete("candidate", candidate)
    if reference["mmlu_total"] != candidate["mmlu_total"]:
        raise ValueError("reference and candidate MMLU totals differ")

    perplexity_relative_change = candidate["perplexity"] / reference["perplexity"] - 1.0
    reference_accuracy = reference["mmlu_correct"] / reference["mmlu_total"]
    candidate_accuracy = candidate["mmlu_correct"] / candidate["mmlu_total"]
    mmlu_accuracy_delta_pp = (candidate_accuracy - reference_accuracy) * 100.0
    repeated_4gram_delta_pp = (
        candidate["repeated_4gram_rate"] - reference["repeated_4gram_rate"]
    ) * 100.0
    distinct_2_delta_pp = (candidate["distinct_2"] - reference["distinct_2"]) * 100.0

    epsilon = 1e-12
    checks = {
        "perplexity": perplexity_relative_change <= 0.005 + epsilon,
        "mmlu_accuracy": mmlu_accuracy_delta_pp >= -1.0 - epsilon,
        "repeated_4gram": repeated_4gram_delta_pp <= 5.0 + epsilon,
        "distinct_2": distinct_2_delta_pp >= -5.0 - epsilon,
        "candidate_determinism": candidate["deterministic"] is True,
        "runtime_cleanliness": candidate["runtime_clean"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "pass": not failed,
        "failed_gates": failed,
        "checks": checks,
        "perplexity_relative_change": perplexity_relative_change,
        "reference_mmlu_accuracy": reference_accuracy,
        "candidate_mmlu_accuracy": candidate_accuracy,
        "mmlu_accuracy_delta_pp": mmlu_accuracy_delta_pp,
        "repeated_4gram_delta_pp": repeated_4gram_delta_pp,
        "distinct_2_delta_pp": distinct_2_delta_pp,
    }
