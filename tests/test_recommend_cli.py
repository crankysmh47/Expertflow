import json
from pathlib import Path

from expertflow.cli.main import main


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_recommend_cli_writes_machine_specific_config(tmp_path: Path) -> None:
    doctor = tmp_path / "doctor.json"
    baseline = tmp_path / "baseline.json"
    profile = tmp_path / "profile.json"
    simulation = tmp_path / "simulation.json"
    output = tmp_path / "recommendation.json"
    write_json(
        doctor,
        {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "gpus": [
                {"index": 0, "name": "Test GPU", "memory_total_mib": 16_000}
            ],
        },
    )
    write_json(
        baseline,
        {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "status": "passed",
            "memory": {"peak_gpu_used_mib": {"0": 8_000}},
        },
    )
    write_json(
        profile,
        {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "profile": {"total_events": 100, "layers": [{"layer_id": 0}]},
        },
    )
    write_json(
        simulation,
        {
            "schema_version": "1.0.0",
            "measurement_kind": "estimated",
            "simulation": {
                "capacity_per_layer": 8,
                "static_hotset": {"hit_rate": 0.4},
                "lru": {"hit_rate": 0.3},
            },
        },
    )

    result = main(
        [
            "recommend",
            "--doctor",
            str(doctor),
            "--baseline",
            str(baseline),
            "--profile",
            str(profile),
            "--simulation",
            str(simulation),
            "--output",
            str(output),
        ]
    )

    recommendation = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert recommendation["verdict"] == "CONDITIONAL"
    assert recommendation["replay"]["policy"] == "static_hotset"
