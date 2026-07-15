import json
from pathlib import Path

import expertflow.cli.main as cli


def test_heldout_breakdown_cli_writes_phase_labeled_report(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "collection-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "breakdown.json"
    captured: dict[str, object] = {}

    def fake_load(
        path, *, phase, evaluation_phase, max_layer, exclude_failed
    ):
        captured.update(
            path=path,
            phase=phase,
            evaluation_phase=evaluation_phase,
            max_layer=max_layer,
            exclude_failed=exclude_failed,
        )
        return (
            ("train-event",),
            ("evaluation-trace",),
            ({"conversation_id": "failed"},),
        )

    def fake_build(
        training,
        evaluations,
        *,
        capacity_per_layer,
        slot_bytes,
        expert_transfer_ms,
    ):
        captured.update(
            training=training,
            evaluations=evaluations,
            capacity=capacity_per_layer,
            slot_bytes=slot_bytes,
            transfer_ms=expert_transfer_ms,
        )
        return {
            "schema_version": "1.0.0",
            "measurement_kind": "estimated_policy_over_measured_routing",
            "per_prompt": [],
            "per_domain": [],
        }

    monkeypatch.setattr(cli, "load_collection_breakdown_inputs", fake_load)
    monkeypatch.setattr(cli, "build_heldout_breakdown", fake_build)

    result = cli.main(
        [
            "heldout-breakdown",
            "--collection-manifest",
            str(manifest),
            "--phase",
            "prefill",
            "--eval-phase",
            "decode",
            "--max-layer",
            "20",
            "--exclude-failed-shards",
            "--capacity",
            "96",
            "--slot-bytes",
            "3346048",
            "--expert-transfer-ms",
            "0.235",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert captured == {
        "path": manifest.resolve(),
        "phase": "prefill",
        "evaluation_phase": "decode",
        "max_layer": 20,
        "exclude_failed": True,
        "training": ("train-event",),
        "evaluations": ("evaluation-trace",),
        "capacity": 96,
        "slot_bytes": 3346048,
        "transfer_ms": 0.235,
    }
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selection"] == {
        "training_phase": "prefill",
        "evaluation_phase": "decode",
        "max_layer": 20,
    }
    assert report["collection_manifest"] == str(manifest.resolve())
    assert report["excluded_shards"] == [{"conversation_id": "failed"}]
