from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle

from expertflow.predictor.dataset import load_pilot_dataset
from expertflow.predictor.models import TransitionPredictor
from expertflow.predictor.runtime_artifact import (
    ArtifactIdentity,
    artifact_identity_payload,
    build_runtime_artifact,
    parse_runtime_artifact,
    predict_runtime_artifact,
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
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _identity(
    manifest: dict[str, object],
    *,
    manifest_sha256: str,
    configuration_sha256: str,
) -> ArtifactIdentity:
    model = manifest.get("model")
    runtime = manifest.get("runtime")
    if not isinstance(model, dict) or not isinstance(model.get("sha256"), str):
        raise ValueError("expanded manifest model identity is missing")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("sha256"), str):
        raise ValueError("expanded manifest runtime identity is missing")
    return ArtifactIdentity(
        model_sha256=model["sha256"],
        runtime_sha256=runtime["sha256"],
        manifest_sha256=manifest_sha256,
        configuration_sha256=configuration_sha256,
    )


def export_bundle(
    *,
    lock_path: Path,
    manifest_path: Path,
    output: Path,
) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_digest = _sha256(manifest_path)
    if lock.get("selected_model") != "b2_source_normalized_separate":
        raise ValueError("selection lock does not name the frozen B2 model")
    if lock.get("selected_candidate_width") != 12:
        raise ValueError("selection lock candidate width is not 12")
    if lock.get("selected_admission_rule") != "observed_support":
        raise ValueError("selection lock admission rule is not observed_support")
    if lock.get("manifest", {}).get("sha256") != manifest_digest:
        raise ValueError("selection lock manifest hash does not match")
    configuration_digest = lock.get("selection_payload_sha256")
    if not isinstance(configuration_digest, str):
        raise ValueError("selection lock configuration hash is missing")
    artifact_record = lock.get("model_artifact")
    if not isinstance(artifact_record, dict) or not isinstance(
        artifact_record.get("path"), str
    ):
        raise ValueError("selection lock model artifact is missing")
    model_path = Path(artifact_record["path"])
    if (
        not model_path.is_file()
        or artifact_record.get("bytes") != model_path.stat().st_size
        or artifact_record.get("sha256") != _sha256(model_path)
    ):
        raise ValueError("frozen B2 model artifact identity does not match")
    with model_path.open("rb") as stream:
        predictor = pickle.load(stream)
    if not isinstance(predictor, TransitionPredictor):
        raise ValueError("frozen model artifact is not a transition predictor")

    artifact_bytes = build_runtime_artifact(
        predictor,
        _identity(
            manifest,
            manifest_sha256=manifest_digest,
            configuration_sha256=configuration_digest,
        ),
    )
    artifact = parse_runtime_artifact(artifact_bytes)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "expertflow-b2-23-24-v1.bin"
    artifact_path.write_bytes(artifact_bytes)

    dataset = load_pilot_dataset(
        manifest_path,
        expected_split_counts=SPLIT_COUNTS,
        expected_domain_counts=DOMAIN_COUNTS,
        require_unique_prompt_hashes=True,
        materialize_splits={"validation"},
    )
    selected_samples = []
    for phase in ("prefill", "decode"):
        phase_samples = [
            sample
            for sample in dataset.validation
            if sample.source_layer == 23
            and sample.target_layer == 24
            and sample.phase == phase
        ][:8]
        if len(phase_samples) != 8:
            raise ValueError(f"validation split lacks eight {phase} layer-23 fixtures")
        selected_samples.extend(phase_samples)

    fixtures = []
    for sample in selected_samples:
        offline_ranking = predictor.rank(sample)
        offline_candidates = tuple(
            expert
            for expert in offline_ranking[:12]
            if predictor.has_support(sample, expert)
        )
        if len(offline_candidates) != 12:
            raise ValueError("frozen observed-support rule returned fewer than 12 candidates")
        runtime_candidates, runtime_scores = predict_runtime_artifact(
            artifact,
            phase=sample.phase,
            source_expert_ids=sample.source_expert_ids,
        )
        if runtime_candidates != offline_candidates:
            raise ValueError("exported artifact does not reproduce offline candidates")
        offline_scores = predictor._scores(sample)
        expected_scores = tuple(float(offline_scores[expert]) for expert in offline_candidates)
        if runtime_scores != expected_scores:
            raise ValueError("exported artifact does not reproduce offline scores")
        fixtures.append(
            {
                "conversation_id": sample.conversation_id,
                "forward_id": sample.forward_id,
                "token_index": sample.token_index,
                "token_id": sample.token_id,
                "phase": sample.phase,
                "source_layer": 23,
                "target_layer": 24,
                "source_expert_ids": list(sample.source_expert_ids),
                "predicted_expert_ids": list(runtime_candidates),
                "predicted_scores": list(runtime_scores),
            }
        )
    fixtures_path = output / "expertflow-b2-23-24-v1-fixtures.json"
    fixtures_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifact_payload_sha256": artifact.payload_sha256,
                "configuration_sha256": configuration_digest,
                "fixtures": fixtures,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = output / "expertflow-b2-23-24-v1.json"
    metadata = {
        "schema_version": "1.0.0",
        "measurement_kind": "frozen_offline_predictor_export",
        "p0_commit": "6bc8eb68",
        "sealed_test_rerun": False,
        "configuration": {
            "family": "B2 transition tables",
            "scoring": "source_normalized",
            "tables": "phase_separated",
            "candidate_width": 12,
            "admission": "observed_support",
            "source_layer": 23,
            "target_layer": 24,
        },
        "identity": artifact_identity_payload(artifact.identity),
        "artifact_payload_sha256": artifact.payload_sha256,
        "artifact": _record(artifact_path),
        "fixtures": _record(fixtures_path),
        "selection_lock": _record(lock_path),
        "manifest": _record(manifest_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    result = export_bundle(
        lock_path=args.selection_lock,
        manifest_path=args.manifest,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
