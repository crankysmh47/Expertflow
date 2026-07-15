from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
from statistics import median
from time import perf_counter_ns
from typing import Sequence

from expertflow.predictor.dataset import PredictionSample, load_pilot_dataset
from expertflow.predictor.learned import LinearPredictor, SharedMlpPredictor, load_learned
from expertflow.predictor.metrics import evaluate_predictions
from expertflow.predictor.models import CopyPredictor, FrequencyPredictor, TransitionPredictor
from expertflow.predictor.shadow import simulate_shadow


WIDTHS = (8, 12, 16)
SEED = 20260716


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _latency(predictor, sample: PredictionSample) -> dict[str, int]:
    if hasattr(predictor, "latency_ns"):
        values = predictor.latency_ns(sample)
    else:
        values = []
        for _ in range(20):
            predictor.rank(sample)
        for _ in range(500):
            start = perf_counter_ns()
            predictor.rank(sample)
            values.append(perf_counter_ns() - start)
    ordered = sorted(values)
    return {
        "sample_count": len(values),
        "p50_ns": int(median(ordered)),
        "p95_ns": ordered[(95 * len(ordered) + 99) // 100 - 1],
    }


def _model_result(predictor, samples: Sequence[PredictionSample]) -> dict[str, object]:
    metrics = evaluate_predictions(samples, predictor)
    rankings = [predictor.rank(sample) for sample in samples]
    return {
        "metrics": metrics,
        "latency_batch1_cpu": _latency(predictor, samples[0]),
        "parameter_count": getattr(predictor, "parameter_count", 0),
        "shadow": {
            str(width): simulate_shadow(samples, rankings, width=width)
            for width in WIDTHS
        },
    }


def _save_model(output: Path, name: str, predictor) -> dict[str, object]:
    suffix = ".pt" if hasattr(predictor, "save") else ".pkl"
    path = output / f"{name}{suffix}"
    if hasattr(predictor, "save"):
        predictor.save(path)
    else:
        with path.open("wb") as stream:
            pickle.dump(predictor, stream)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_model(record: dict[str, object]):
    path = Path(str(record["path"]))
    if path.suffix == ".pt":
        return load_learned(path)
    with path.open("rb") as stream:
        return pickle.load(stream)


def _choose(validation: dict[str, dict[str, object]]) -> tuple[str, int, bool]:
    frequency = validation["b1_frequency"]["metrics"]
    frequency_recall = float(frequency["recall_at_16"])
    candidates = ("b2_transition", "b3_linear", "b4_shared_mlp")
    order = {name: index for index, name in enumerate(candidates)}
    selected = max(candidates, key=lambda name: (
        float(validation[name]["metrics"]["recall_at_16"]),
        float(validation[name]["metrics"]["recall_at_8"]),
        -order[name],
    ))
    shadows = validation[selected]["shadow"]
    width = max(WIDTHS, key=lambda candidate: (
        int(shadows[str(candidate)]["ready_improvement_over_reactive"]) -
        int(shadows[str(candidate)]["eviction_regret"]),
        -int(shadows[str(candidate)]["wasted_predicted_bytes"]),
        -candidate,
    ))
    promising = (
        float(validation[selected]["metrics"]["recall_at_16"]) > frequency_recall and
        int(shadows[str(width)]["ready_improvement_over_reactive"]) > 0
    )
    return selected, width, promising


def _fit(manifest: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection-lock.json").exists():
        raise ValueError("selection lock already exists; refusing to refit the frozen pilot")
    dataset = load_pilot_dataset(manifest)
    models = {
        "b0_copy": CopyPredictor(),
        "b1_frequency": FrequencyPredictor.fit(dataset.train),
        "b2_transition": TransitionPredictor.fit(dataset.train),
        "b3_linear": LinearPredictor.fit(dataset.train),
        "b4_shared_mlp": SharedMlpPredictor.fit(dataset.train),
    }
    validation: dict[str, dict[str, object]] = {}
    artifacts: dict[str, dict[str, object]] = {}
    for name, predictor in models.items():
        validation[name] = _model_result(predictor, dataset.validation)
        artifacts[name] = _save_model(output, name, predictor)
        validation[name]["artifact"] = artifacts[name]
    selected, width, promising = _choose(validation)
    metrics = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured_routing_offline_prediction_and_simulated_shadow",
        "pilot_scope": "14-conversation small-pilot feasibility only",
        "split": "validation",
        "conversation_ids": dataset.conversation_ids["validation"],
        "models": validation,
    }
    _write_json(output / "validation-metrics.json", metrics)
    lock = {
        "schema_version": "1.0.0",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest.resolve()), "sha256": _sha256(manifest)},
        "frozen_split": {key: list(value) for key, value in dataset.conversation_ids.items()},
        "selected_model": selected,
        "selected_candidate_width": width,
        "evaluated_candidate_widths": list(WIDTHS),
        "seed": SEED,
        "feature_contract": "128 source + 30 target-layer one-hot + decode bit + 128 causal previous-target",
        "weights": "binary because canonical pilot selected_expert_weights are unavailable",
        "model_artifact": artifacts[selected],
        "validation_promising": promising,
        "test_opened": False,
    }
    _write_json(output / "selection-lock.json", lock)
    return 0


def _test(manifest: Path, output: Path) -> int:
    lock_path = output / "selection-lock.json"
    if not lock_path.is_file():
        raise ValueError("selection lock is required before opening the frozen test split")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_opened") is True or (output / "test-metrics.json").exists():
        raise ValueError("frozen test split has already been evaluated")
    if lock["manifest"]["sha256"] != _sha256(manifest):
        raise ValueError("selection lock manifest hash does not match")
    dataset = load_pilot_dataset(manifest)
    if lock["frozen_split"]["test"] != list(dataset.conversation_ids["test"]):
        raise ValueError("frozen test conversation IDs changed")
    predictor = _load_model(lock["model_artifact"])
    result = _model_result(predictor, dataset.test)
    metrics = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured_routing_offline_prediction_and_simulated_shadow",
        "pilot_scope": "14-conversation small-pilot feasibility only",
        "split": "test",
        "conversation_ids": dataset.conversation_ids["test"],
        "selected_model": lock["selected_model"],
        "selected_candidate_width": lock["selected_candidate_width"],
        "result": result,
    }
    _write_json(output / "test-metrics.json", metrics)
    lock["test_opened"] = True
    lock["test_opened_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(lock_path, lock)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "test"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output / "ledger.jsonl"
    _append(ledger, {"at": datetime.now(timezone.utc).isoformat(), "event": "command_start",
                     "command": args.command, "manifest": str(args.manifest.resolve())})
    try:
        result = _fit(args.manifest, args.output) if args.command == "fit" else _test(args.manifest, args.output)
    except Exception as error:
        _append(ledger, {"at": datetime.now(timezone.utc).isoformat(), "event": "command_failed",
                         "command": args.command, "error": str(error)})
        raise
    _append(ledger, {"at": datetime.now(timezone.utc).isoformat(), "event": "command_end",
                     "command": args.command, "return_code": result})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
