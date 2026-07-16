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
EXPANDED_SPLIT_COUNTS = {"train": 60, "validation": 12, "test": 12}
EXPANDED_DOMAINS = (
    "general_instruction",
    "code",
    "math_reasoning",
    "translation_multilingual",
    "structured_output",
    "topic_shift",
)
EXPANDED_DOMAIN_COUNTS = {
    "train": {domain: 10 for domain in EXPANDED_DOMAINS},
    "validation": {domain: 2 for domain in EXPANDED_DOMAINS},
    "test": {domain: 2 for domain in EXPANDED_DOMAINS},
}
B2_CONFIGS = (
    ("b2_raw_count_pooled", "raw_count", "pooled"),
    ("b2_raw_count_separate", "raw_count", "separate"),
    ("b2_source_normalized_pooled", "source_normalized", "pooled"),
    ("b2_source_normalized_separate", "source_normalized", "separate"),
)


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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _admitted_rankings(
    predictor,
    samples: Sequence[PredictionSample],
    rankings: Sequence[tuple[int, ...]],
    *,
    width: int,
    admission_rule: str,
) -> list[tuple[int, ...]]:
    if admission_rule == "all_ranked":
        return [tuple(ranking[:width]) for ranking in rankings]
    if admission_rule != "observed_support" or not hasattr(predictor, "has_support"):
        raise ValueError("observed_support admission requires a transition predictor")
    return [
        tuple(expert for expert in ranking[:width] if predictor.has_support(sample, expert))
        for sample, ranking in zip(samples, rankings, strict=True)
    ]


def _expanded_b2_result(predictor, samples: Sequence[PredictionSample]) -> dict[str, object]:
    metrics = evaluate_predictions(samples, predictor)
    rankings = [predictor.rank(sample) for sample in samples]
    policies: dict[str, dict[str, object]] = {}
    for admission_rule in ("all_ranked", "observed_support"):
        policies[admission_rule] = {}
        for width in WIDTHS:
            admitted = _admitted_rankings(
                predictor, samples, rankings, width=width, admission_rule=admission_rule
            )
            policies[admission_rule][str(width)] = simulate_shadow(
                samples,
                rankings,
                admitted_rankings=admitted,
                width=width,
            )
    return {
        "metrics": metrics,
        "latency_batch1_cpu": _latency(predictor, samples[0]),
        "parameter_count": 0,
        "policy_shadows": policies,
    }


def _choose_expanded(
    validation: dict[str, dict[str, object]],
    *,
    widths: Sequence[int] = WIDTHS,
) -> dict[str, object]:
    weighting_order = {"raw_count": 0, "source_normalized": 1}
    phase_order = {"pooled": 0, "separate": 1}
    admission_order = {"observed_support": 0, "all_ranked": 1}
    candidates: list[tuple[tuple[float, ...], dict[str, object]]] = []
    for model_name, weighting, phase_mode in B2_CONFIGS:
        result = validation[model_name]
        metrics = result["metrics"]
        policies = result["policy_shadows"]
        for admission_rule in ("observed_support", "all_ranked"):
            for width in widths:
                shadow = policies[admission_rule][str(width)]
                net_gain = (
                    int(shadow["ready_improvement_over_reactive"])
                    - int(shadow["eviction_regret"])
                )
                recall = float(metrics[f"recall_at_{width}"])
                key = (
                    float(net_gain),
                    -float(shadow["wasted_predicted_bytes"]),
                    recall,
                    -float(weighting_order[weighting]),
                    -float(phase_order[phase_mode]),
                    -float(admission_order[admission_rule]),
                    -float(width),
                )
                candidates.append((key, {
                    "model": model_name,
                    "candidate_width": width,
                    "admission_rule": admission_rule,
                }))
    return max(candidates, key=lambda row: row[0])[1]


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


def _load_expanded_dataset(manifest: Path, *, materialize_splits: set[str]):
    return load_pilot_dataset(
        manifest,
        expected_split_counts=EXPANDED_SPLIT_COUNTS,
        expected_domain_counts=EXPANDED_DOMAIN_COUNTS,
        require_unique_prompt_hashes=True,
        materialize_splits=materialize_splits,
    )


def _selection_payload(lock: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in lock.items()
        if key not in {"selection_payload_sha256", "test_opened", "test_opened_at"}
    }


def _write_artifact_index(output: Path) -> None:
    records = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "artifact-index.json":
            continue
        records.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    _write_json(output / "artifact-index.json", {"schema_version": "1.0.0", "artifacts": records})


def _deduplication_provenance(manifest_payload: dict[str, object]) -> dict[str, object]:
    record = manifest_payload.get("frozen_corpus")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("expanded manifest frozen-corpus provenance is missing")
    path = Path(record["path"])
    if not path.is_file():
        raise ValueError("expanded frozen-corpus definition is missing")
    if record.get("bytes") != path.stat().st_size or record.get("sha256") != _sha256(path):
        raise ValueError("expanded frozen-corpus definition hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = payload.get("deduplication_policy")
    if not isinstance(policy, dict) or not policy:
        raise ValueError("expanded frozen-corpus deduplication policy is missing")
    return {"frozen_corpus": record, "policy": policy}


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


def _fit_expanded(manifest: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection-lock.json").exists():
        raise ValueError("selection lock already exists; refusing to refit the expanded corpus")
    dataset = _load_expanded_dataset(manifest, materialize_splits={"train", "validation"})
    models = {
        "b0_copy": CopyPredictor(),
        "b1_frequency": FrequencyPredictor.fit(dataset.train),
        **{
            name: TransitionPredictor.fit(
                dataset.train, weighting=weighting, phase_mode=phase_mode
            )
            for name, weighting, phase_mode in B2_CONFIGS
        },
        "b3_linear": LinearPredictor.fit(dataset.train),
        "b4_shared_mlp": SharedMlpPredictor.fit(dataset.train),
    }
    validation: dict[str, dict[str, object]] = {}
    artifacts: dict[str, dict[str, object]] = {}
    b2_names = {name for name, _, _ in B2_CONFIGS}
    for name, predictor in models.items():
        result = (
            _expanded_b2_result(predictor, dataset.validation)
            if name in b2_names
            else _model_result(predictor, dataset.validation)
        )
        artifact = _save_model(output, name, predictor)
        result["artifact"] = artifact
        validation[name] = result
        artifacts[name] = artifact

    selected = _choose_expanded(validation)
    selected_b2 = str(selected["model"])
    b2_recall_8 = float(validation[selected_b2]["metrics"]["recall_at_8"])
    b2_p95 = int(validation[selected_b2]["latency_batch1_cpu"]["p95_ns"])
    eligible_learned = [
        name
        for name in ("b3_linear", "b4_shared_mlp")
        if (
            float(validation[name]["metrics"]["recall_at_8"]) >= b2_recall_8 + 0.02
            and int(validation[name]["latency_batch1_cpu"]["p95_ns"]) <= 2 * b2_p95
        )
    ]
    if eligible_learned:
        learned_name = max(
            eligible_learned,
            key=lambda name: (
                float(validation[name]["metrics"]["recall_at_8"]),
                -int(validation[name]["latency_batch1_cpu"]["p95_ns"]),
                name == "b3_linear",
            ),
        )
        learned_shadows = validation[learned_name]["shadow"]
        learned_width = max(
            WIDTHS,
            key=lambda width: (
                int(learned_shadows[str(width)]["ready_improvement_over_reactive"])
                - int(learned_shadows[str(width)]["eviction_regret"]),
                -int(learned_shadows[str(width)]["wasted_predicted_bytes"]),
                float(validation[learned_name]["metrics"][f"recall_at_{width}"]),
                -width,
            ),
        )
        selected = {
            "model": learned_name,
            "candidate_width": learned_width,
            "admission_rule": "all_ranked",
        }

    validation_metrics = {
        "schema_version": "2.0.0",
        "measurement_kind": "measured_routing_offline_prediction_and_simulated_shadow",
        "scope": "frozen 84-conversation canonical expanded corpus",
        "split": "validation",
        "conversation_ids": dataset.conversation_ids["validation"],
        "sample_count": len(dataset.validation),
        "models": validation,
        "fixed_selection_rule": {
            "b2_search": "four bounded configurations x widths 8/12/16 x two admission rules",
            "objective": "net ready gain after eviction regret, waste, recall, simplicity",
            "learned_override": "recall@8 >= best B2 + 0.02 and p95 <= 2x best B2",
        },
        "selected": selected,
    }
    validation_path = output / "validation-metrics.json"
    _write_json(validation_path, validation_metrics)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    lock: dict[str, object] = {
        "schema_version": "2.0.0",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest.resolve()), "sha256": _sha256(manifest)},
        "deduplication": _deduplication_provenance(manifest_payload),
        "frozen_split": {key: list(value) for key, value in dataset.conversation_ids.items()},
        "frozen_split_counts": EXPANDED_SPLIT_COUNTS,
        "frozen_domain_counts": EXPANDED_DOMAIN_COUNTS,
        "selected_model": selected["model"],
        "selected_b2_configuration": (
            selected["model"] if str(selected["model"]).startswith("b2_") else selected_b2
        ),
        "selected_candidate_width": selected["candidate_width"],
        "selected_admission_rule": selected["admission_rule"],
        "evaluated_candidate_widths": list(WIDTHS),
        "evaluated_b2_configurations": [name for name, _, _ in B2_CONFIGS],
        "seed": SEED,
        "feature_contract": "128 source + 30 target-layer one-hot + decode bit + 128 causal previous-target",
        "weights": "binary because canonical selected_expert_weights are unavailable",
        "model_artifact": artifacts[str(selected["model"])],
        "validation_metrics": {
            "path": str(validation_path.resolve()),
            "bytes": validation_path.stat().st_size,
            "sha256": _sha256(validation_path),
        },
        "test_opened": False,
    }
    lock["selection_payload_sha256"] = _canonical_sha256(_selection_payload(lock))
    _write_json(output / "selection-lock.json", lock)
    _write_artifact_index(output)
    return 0


def _test_expanded(manifest: Path, output: Path) -> int:
    lock_path = output / "selection-lock.json"
    test_path = output / "test-metrics.json"
    if not lock_path.is_file():
        raise ValueError("selection lock is required before opening the expanded test split")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_opened") is True or test_path.exists():
        raise ValueError("expanded test split has already been evaluated")
    if lock.get("selection_payload_sha256") != _canonical_sha256(_selection_payload(lock)):
        raise ValueError("selection lock payload hash does not match")
    if lock["manifest"]["sha256"] != _sha256(manifest):
        raise ValueError("selection lock manifest hash does not match")
    artifact = lock["model_artifact"]
    if artifact["sha256"] != _sha256(Path(artifact["path"])):
        raise ValueError("selected model artifact hash does not match")
    dataset = _load_expanded_dataset(manifest, materialize_splits={"test"})
    if lock["frozen_split"]["test"] != list(dataset.conversation_ids["test"]):
        raise ValueError("frozen expanded test conversation IDs changed")
    predictor = _load_model(artifact)
    samples = dataset.test
    metrics = evaluate_predictions(samples, predictor)
    rankings = [predictor.rank(sample) for sample in samples]
    width = int(lock["selected_candidate_width"])
    admission = str(lock["selected_admission_rule"])
    if admission == "observed_support":
        admitted = _admitted_rankings(
            predictor, samples, rankings, width=width, admission_rule=admission
        )
    else:
        admitted = [tuple(ranking[:width]) for ranking in rankings]
    result = {
        "metrics": metrics,
        "latency_batch1_cpu": _latency(predictor, samples[0]),
        "parameter_count": getattr(predictor, "parameter_count", 0),
        "selected_shadow": simulate_shadow(
            samples,
            rankings,
            admitted_rankings=admitted,
            width=width,
        ),
    }
    test_metrics = {
        "schema_version": "2.0.0",
        "measurement_kind": "measured_routing_offline_prediction_and_simulated_shadow",
        "scope": "single sealed-test evaluation on frozen 84-conversation canonical expanded corpus",
        "split": "test",
        "conversation_ids": dataset.conversation_ids["test"],
        "sample_count": len(samples),
        "selected_model": lock["selected_model"],
        "selected_b2_configuration": lock["selected_b2_configuration"],
        "selected_candidate_width": width,
        "selected_admission_rule": admission,
        "result": result,
    }
    _write_json(test_path, test_metrics)
    lock["test_opened"] = True
    lock["test_opened_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(lock_path, lock)
    _write_artifact_index(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "test"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output / "ledger.jsonl"
    _append(ledger, {"at": datetime.now(timezone.utc).isoformat(), "event": "command_start",
                     "command": args.command, "manifest": str(args.manifest.resolve()),
                     "expanded": args.expanded})
    try:
        if args.expanded:
            result = (
                _fit_expanded(args.manifest, args.output)
                if args.command == "fit"
                else _test_expanded(args.manifest, args.output)
            )
        else:
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
