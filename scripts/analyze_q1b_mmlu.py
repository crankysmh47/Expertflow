from __future__ import annotations

import argparse
import json
from pathlib import Path

from expertflow.quality.mmlu import mmlu_gate_pass


def _identity(item: dict[str, object]) -> tuple[object, ...]:
    return item["index"], item["subject"], item["row_id"], item["selection_sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare paired Q1b MMLU results")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    repeat = json.loads(args.candidate_repeat.read_text(encoding="utf-8"))
    if not (reference["total"] == candidate["total"] == repeat["total"]):
        raise ValueError("MMLU totals differ")
    for index, records in enumerate(zip(reference["items"], candidate["items"], repeat["items"], strict=True)):
        if len({_identity(record) for record in records}) != 1:
            raise ValueError(f"MMLU identity mismatch at item {index}")
    changed = [
        reference_item["index"]
        for reference_item, candidate_item in zip(reference["items"], candidate["items"], strict=True)
        if reference_item["prediction"] != candidate_item["prediction"]
    ]
    repeat_changed = [
        candidate_item["index"]
        for candidate_item, repeat_item in zip(candidate["items"], repeat["items"], strict=True)
        if (
            candidate_item["prediction"], candidate_item["tokens"], candidate_item["content"]
        )
        != (repeat_item["prediction"], repeat_item["tokens"], repeat_item["content"])
    ]
    subjects: dict[str, dict[str, int | float]] = {}
    disagreements: list[dict[str, object]] = []
    for reference_item, candidate_item in zip(reference["items"], candidate["items"], strict=True):
        subject = str(reference_item["subject"])
        subject_result = subjects.setdefault(
            subject, {"total": 0, "reference_correct": 0, "candidate_correct": 0}
        )
        subject_result["total"] += 1
        subject_result["reference_correct"] += int(bool(reference_item["correct"]))
        subject_result["candidate_correct"] += int(bool(candidate_item["correct"]))
        if reference_item["prediction"] != candidate_item["prediction"]:
            disagreements.append(
                {
                    "index": reference_item["index"],
                    "subject": subject,
                    "row_id": reference_item["row_id"],
                    "answer": reference_item["answer"],
                    "reference_prediction": reference_item["prediction"],
                    "candidate_prediction": candidate_item["prediction"],
                }
            )
    for subject_result in subjects.values():
        total = int(subject_result["total"])
        subject_result["reference_accuracy"] = int(subject_result["reference_correct"]) / total
        subject_result["candidate_accuracy"] = int(subject_result["candidate_correct"]) / total
        subject_result["delta_percentage_points"] = (
            float(subject_result["candidate_accuracy"]) - float(subject_result["reference_accuracy"])
        ) * 100.0
    result = {
        "paired_items": reference["total"],
        "reference_correct": reference["correct"],
        "candidate_correct": candidate["correct"],
        "candidate_repeat_correct": repeat["correct"],
        "accuracy_delta_percentage_points": (candidate["accuracy"] - reference["accuracy"]) * 100.0,
        "prediction_changes_reference_to_candidate": changed,
        "candidate_repeat_changes": repeat_changed,
        "candidate_deterministic": not repeat_changed and candidate["correct"] == repeat["correct"],
        "subjects": subjects,
        "disagreements": disagreements,
        "gate_pass": mmlu_gate_pass((candidate["accuracy"] - reference["accuracy"]) * 100.0)
        and not repeat_changed,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
