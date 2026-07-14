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


def test_recommend_cli_accepts_capacity_curve(tmp_path: Path) -> None:
    doctor = tmp_path / "doctor.json"
    baseline = tmp_path / "baseline.json"
    profile = tmp_path / "profile.json"
    simulation = tmp_path / "simulation.json"
    curve = tmp_path / "curve.json"
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
    write_json(
        curve,
        {
            "schema_version": "1.0.0",
            "measurement_kind": "estimated",
            "fit_scope": "in_sample",
            "event_count": 200,
            "expert_demand_count": 1_600,
            "layer_ids": [0],
            "router_top_k": 8,
            "slot_bytes": 100,
            "expert_transfer_ms": 0.2,
            "points": [
                {
                    "capacity_per_layer": 16,
                    "projected_cache_bytes": 1_600,
                    "static_hotset": {
                        "hit_rate": 0.6,
                        "estimated_serial_h2d_ms_per_layer_sweep": 0.5,
                    },
                    "lru": {
                        "hit_rate": 0.5,
                        "estimated_serial_h2d_ms_per_layer_sweep": 0.7,
                    },
                }
            ],
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
            "--capacity-curve",
            str(curve),
            "--output",
            str(output),
        ]
    )

    recommendation = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert recommendation["replay"]["capacity_per_layer"] == 16
    assert recommendation["sources"]["capacity_curve"] == str(curve.resolve())
