from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
from statistics import median
from time import perf_counter_ns
from typing import Sequence

from expertflow.predictor.temporal_dataset import TemporalSample, load_temporal_dataset
from expertflow.predictor.temporal_metrics import evaluate_temporal_predictions
from expertflow.predictor.temporal_models import (
    TemporalCombinedPredictor,
    TemporalCopyPredictor,
    TemporalSessionFrequencyPredictor,
    TemporalTransitionPredictor,
    rank_temporal_samples,
)
from expertflow.predictor.temporal_shadow import simulate_temporal_shadow


WIDTHS = (8, 12, 16)
SPLIT_COUNTS = {"train": 60, "validation": 12, "test": 12}
DOMAINS = (
    "general_instruction",
    "code",
    "math_reasoning",
    "translation_multilingual",
    "structured_output",
    "topic_shift",
)
DOMAIN_COUNTS = {
    "train": {domain: 10 for domain in DOMAINS},
    "validation": {domain: 2 for domain in DOMAINS},
    "test": {domain: 2 for domain in DOMAINS},
}
COMBINED_WEIGHTS = (
    (0.50, 0.25, 0.25),
    (0.50, 0.40, 0.10),
    (0.60, 0.20, 0.20),
    (0.70, 0.20, 0.10),
)
POLICY_ORDER = {
    "t0.0_copy": 0,
    "t0.1_session_frequency": 1,
    "t0.2_transition": 2,
    "t0.3_combined": 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_dataset(manifest: Path, materialize: set[str]):
    return load_temporal_dataset(
        manifest,
        expected_split_counts=SPLIT_COUNTS,
        expected_domain_counts=DOMAIN_COUNTS,
        require_unique_prompt_hashes=True,
        materialize_splits=materialize,
    )


def _latency(predictor, sample: TemporalSample) -> dict[str, int]:
    counts = Counter(sample.source_expert_ids)
    for _ in range(20):
        predictor.rank(sample, counts)
    values = []
    for _ in range(500):
        start = perf_counter_ns()
        predictor.rank(sample, counts)
        values.append(perf_counter_ns() - start)
    ordered = sorted(values)
    return {
        "sample_count": len(ordered),
        "p50_ns": int(median(ordered)),
        "p95_ns": ordered[(95 * len(ordered) + 99) // 100 - 1],
    }


def _candidate_rows(samples: Sequence[TemporalSample], predictors) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for predictor in predictors:
        rankings = rank_temporal_samples(samples, predictor)
        metrics = evaluate_temporal_predictions(samples, rankings)
        latency = _latency(predictor, samples[0])
        for width in WIDTHS:
            rows.append({
                "policy": predictor.name,
                "candidate_width": width,
                "weights": list(predictor.weights) if hasattr(predictor, "weights") else None,
                "metrics": metrics,
                "latency_batch1_cpu": latency,
                "shadow": simulate_temporal_shadow(samples, rankings, width=width),
            })
    return rows


def _choose_validation_candidate(candidates: Sequence[dict[str, object]]) -> dict[str, object]:
    def key(candidate: dict[str, object]) -> tuple[float, ...]:
        shadow = candidate["shadow"]
        metrics = candidate["metrics"]
        width = int(candidate["candidate_width"])
        net = int(shadow["ready_improvement_over_reactive"]) - int(shadow["eviction_regret"])
        return (
            float(net),
            -float(shadow["wasted_predicted_bytes"]),
            float(metrics[f"recall_at_{width}"]),
            -float(candidate["latency_batch1_cpu"]["p95_ns"]),
            -float(width),
            -float(POLICY_ORDER[str(candidate["policy"])]),
        )
    return max(candidates, key=key)


def _selection_payload(lock: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in lock.items()
        if key not in {"selection_payload_sha256", "test_opened", "test_opened_at"}
    }


def _artifact_index(output: Path) -> None:
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact-index.json":
            artifacts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_json(output / "artifact-index.json", {"schema_version": "1.0.0", "artifacts": artifacts})


def _fit(manifest: Path, output: Path) -> int:
    lock_path = output / "selection-lock.json"
    if lock_path.exists():
        raise ValueError("selection lock already exists; refusing to refit temporal predictor")
    dataset = _load_dataset(manifest, {"train", "validation"})
    transition = TemporalTransitionPredictor.fit(dataset.train)
    predictors = [
        TemporalCopyPredictor(),
        TemporalSessionFrequencyPredictor(),
        transition,
        *(TemporalCombinedPredictor(transition, weights) for weights in COMBINED_WEIGHTS),
    ]
    candidates = _candidate_rows(dataset.validation, predictors)
    selected = _choose_validation_candidate(candidates)
    selected_predictor = next(
        predictor for predictor in predictors
        if predictor.name == selected["policy"]
        and (
            selected["weights"] is None
            or list(getattr(predictor, "weights", ())) == selected["weights"]
        )
    )
    artifact_path = output / "selected-temporal-predictor.pkl"
    with artifact_path.open("wb") as stream:
        pickle.dump(selected_predictor, stream)
    validation_path = output / "validation-metrics.json"
    _write_json(validation_path, {
        "schema_version": "1.0.0",
        "measurement_kind": "measured_offline_temporal_prediction_and_simulated_shadow",
        "scope": "decode-only layer-24 next-token validation on frozen expanded corpus",
        "conversation_ids": dataset.conversation_ids["validation"],
        "sample_count": len(dataset.validation),
        "fixed_policy_grid": {
            "policies": ["t0.0_copy", "t0.1_session_frequency", "t0.2_transition", "t0.3_combined"],
            "combined_weights": [list(weights) for weights in COMBINED_WEIGHTS],
            "candidate_widths": list(WIDTHS),
        },
        "candidates": candidates,
        "selected": {
            "policy": selected["policy"],
            "weights": selected["weights"],
            "candidate_width": selected["candidate_width"],
        },
    })
    lock: dict[str, object] = {
        "schema_version": "1.0.0",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest.resolve()), "sha256": _sha256(manifest)},
        "frozen_split": {key: list(value) for key, value in dataset.conversation_ids.items()},
        "frozen_split_counts": SPLIT_COUNTS,
        "frozen_domain_counts": DOMAIN_COUNTS,
        "join_contract": "same conversation/request/turn; decode; layer 24; exact +1 forward and token index",
        "adjacent_layer_test_used_for_selection": False,
        "selected_policy": selected["policy"],
        "selected_weights": selected["weights"],
        "selected_candidate_width": selected["candidate_width"],
        "evaluated_candidate_widths": list(WIDTHS),
        "evaluated_combined_weights": [list(weights) for weights in COMBINED_WEIGHTS],
        "model_artifact": _record(artifact_path),
        "validation_metrics": _record(validation_path),
        "test_opened": False,
    }
    lock["selection_payload_sha256"] = _canonical_sha256(_selection_payload(lock))
    _write_json(lock_path, lock)
    _artifact_index(output)
    return 0


def _test(manifest: Path, output: Path) -> int:
    lock_path = output / "selection-lock.json"
    test_path = output / "test-metrics.json"
    if not lock_path.is_file():
        raise ValueError("selection lock is required before opening temporal test split")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_opened") is True or test_path.exists():
        raise ValueError("temporal test split has already been evaluated")
    if lock.get("selection_payload_sha256") != _canonical_sha256(_selection_payload(lock)):
        raise ValueError("selection lock payload hash does not match")
    if lock["manifest"]["sha256"] != _sha256(manifest):
        raise ValueError("selection lock manifest hash does not match")
    artifact_path = Path(lock["model_artifact"]["path"])
    if lock["model_artifact"]["sha256"] != _sha256(artifact_path):
        raise ValueError("selected temporal model artifact hash does not match")
    dataset = _load_dataset(manifest, {"test"})
    if lock["frozen_split"]["test"] != list(dataset.conversation_ids["test"]):
        raise ValueError("frozen temporal test identities changed")
    with artifact_path.open("rb") as stream:
        predictor = pickle.load(stream)
    rankings = rank_temporal_samples(dataset.test, predictor)
    metrics = evaluate_temporal_predictions(dataset.test, rankings)
    width = int(lock["selected_candidate_width"])
    _write_json(test_path, {
        "schema_version": "1.0.0",
        "measurement_kind": "measured_offline_temporal_prediction_and_simulated_shadow",
        "scope": "single sealed decode-only layer-24 temporal test evaluation",
        "conversation_ids": dataset.conversation_ids["test"],
        "sample_count": len(dataset.test),
        "selected_policy": lock["selected_policy"],
        "selected_weights": lock["selected_weights"],
        "selected_candidate_width": width,
        "metrics": metrics,
        "latency_batch1_cpu": _latency(predictor, dataset.test[0]),
        "shadow": simulate_temporal_shadow(dataset.test, rankings, width=width),
    })
    lock["test_opened"] = True
    lock["test_opened_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(lock_path, lock)
    _artifact_index(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "test"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output / "ledger.jsonl"
    _append(ledger, {
        "at": datetime.now(timezone.utc).isoformat(),
        "event": "command_start",
        "command": args.command,
        "manifest": str(args.manifest.resolve()),
    })
    try:
        result = _fit(args.manifest, args.output) if args.command == "fit" else _test(args.manifest, args.output)
    except Exception as error:
        _append(ledger, {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": "command_failed",
            "command": args.command,
            "error": str(error),
        })
        raise
    _append(ledger, {
        "at": datetime.now(timezone.utc).isoformat(),
        "event": "command_end",
        "command": args.command,
        "return_code": result,
    })
    return result


if __name__ == "__main__":
    raise SystemExit(main())

