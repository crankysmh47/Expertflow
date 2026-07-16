from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import pickle

from expertflow.predictor.temporal_dataset import load_temporal_dataset
from expertflow.predictor.temporal_models import TemporalCombinedPredictor
from expertflow.predictor.temporal_runtime_artifact import (
    TemporalArtifactIdentity,
    build_temporal_runtime_artifact,
    parse_temporal_runtime_artifact,
    predict_temporal_runtime_artifact,
)


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


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def export_temporal_bundle(
    *,
    lock_path: Path,
    manifest_path: Path,
    output: Path,
) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        lock.get("selected_policy") != "t0.3_combined"
        or lock.get("selected_weights") != [0.5, 0.4, 0.1]
        or lock.get("selected_candidate_width") != 16
        or lock.get("test_opened") is not True
        or lock.get("manifest", {}).get("sha256") != _sha256(manifest_path)
    ):
        raise ValueError("selection lock is not the frozen T0 temporal configuration")
    artifact_record = lock.get("model_artifact")
    if not isinstance(artifact_record, dict) or not isinstance(artifact_record.get("path"), str):
        raise ValueError("temporal selection lock model artifact is missing")
    model_path = Path(artifact_record["path"])
    if not model_path.is_file() or artifact_record.get("sha256") != _sha256(model_path):
        raise ValueError("temporal selected model artifact identity does not match")
    with model_path.open("rb") as stream:
        predictor = pickle.load(stream)
    if (
        not isinstance(predictor, TemporalCombinedPredictor)
        or predictor.weights != (0.5, 0.4, 0.1)
    ):
        raise ValueError("selected model is not the frozen temporal combined predictor")
    model = manifest.get("model")
    runtime = manifest.get("runtime")
    if not isinstance(model, dict) or not isinstance(model.get("sha256"), str):
        raise ValueError("manifest model identity is missing")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("sha256"), str):
        raise ValueError("manifest runtime identity is missing")
    identity = TemporalArtifactIdentity(
        model["sha256"],
        runtime["sha256"],
        _sha256(manifest_path),
        lock["selection_payload_sha256"],
    )
    artifact_bytes = build_temporal_runtime_artifact(predictor.transition, identity)
    artifact = parse_temporal_runtime_artifact(artifact_bytes)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "expertflow-temporal-layer24-v1.bin"
    artifact_path.write_bytes(artifact_bytes)

    dataset = load_temporal_dataset(
        manifest_path,
        expected_split_counts=SPLIT_COUNTS,
        expected_domain_counts=DOMAIN_COUNTS,
        require_unique_prompt_hashes=True,
        materialize_splits={"validation"},
    )
    fixtures = []
    current_conversation = None
    session_counts: Counter[int] = Counter()
    per_conversation = Counter()
    for sample in dataset.validation:
        if sample.conversation_id != current_conversation:
            current_conversation = sample.conversation_id
            session_counts.clear()
        before = [session_counts[index] for index in range(128)]
        candidates, scores, updated = predict_temporal_runtime_artifact(
            artifact,
            source_expert_ids=sample.source_expert_ids,
            session_counts=session_counts,
        )
        session_counts = updated
        offline = predictor.rank(sample, session_counts)
        if candidates != offline[:16]:
            raise ValueError("temporal runtime artifact does not reproduce offline candidate order")
        if per_conversation[sample.conversation_id] < 4:
            fixtures.append({
                "conversation_id": sample.conversation_id,
                "source_forward_id": sample.source_forward_id,
                "target_forward_id": sample.target_forward_id,
                "source_expert_ids": list(sample.source_expert_ids),
                "session_counts_before": before,
                "session_counts_after": [session_counts[index] for index in range(128)],
                "predicted_expert_ids": list(candidates),
                "predicted_scores": list(scores),
            })
            per_conversation[sample.conversation_id] += 1
    fixtures_path = output / "expertflow-temporal-layer24-v1-fixtures.json"
    fixtures_path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "artifact_payload_sha256": artifact.payload_sha256,
        "configuration_sha256": identity.configuration_sha256,
        "fixtures": fixtures,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path = output / "expertflow-temporal-layer24-v1.json"
    metadata_path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "measurement_kind": "frozen_offline_temporal_predictor_export",
        "configuration": {
            "layer": 24,
            "phase": "decode",
            "candidate_width": 16,
            "weights": [0.5, 0.4, 0.1],
            "tie_break": "ascending_expert_id",
        },
        "identity": {
            "model_sha256": identity.model_sha256,
            "runtime_sha256": identity.runtime_sha256,
            "manifest_sha256": identity.manifest_sha256,
            "configuration_sha256": identity.configuration_sha256,
        },
        "artifact_payload_sha256": artifact.payload_sha256,
        "artifact": _record(artifact_path),
        "fixtures": _record(fixtures_path),
        "selection_lock": _record(lock_path),
        "manifest": _record(manifest_path),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifact": _record(artifact_path),
        "fixtures": _record(fixtures_path),
        "metadata": _record(metadata_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_temporal_bundle(
        lock_path=args.selection_lock,
        manifest_path=args.manifest,
        output=args.output,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
