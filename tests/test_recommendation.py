import pytest

from expertflow.recommendation import RecommendationInputError, build_recommendation


def evidence() -> tuple[dict[str, object], ...]:
    doctor = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "gpus": [
            {
                "index": 0,
                "name": "Test GPU",
                "memory_total_mib": 16_000,
            }
        ],
    }
    baseline = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "status": "passed",
        "memory": {"peak_gpu_used_mib": {"0": 8_000}},
    }
    profile = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "profile": {"total_events": 1_350, "layers": [{"layer_id": 0}]},
    }
    simulation = {
        "schema_version": "1.0.0",
        "measurement_kind": "estimated",
        "simulation": {
            "capacity_per_layer": 8,
            "static_hotset": {"policy": "static_hotset", "hit_rate": 0.36},
            "lru": {"policy": "lru", "hit_rate": 0.32},
            "reactive": {"policy": "reactive", "hit_rate": 0.0},
        },
    }
    return doctor, baseline, profile, simulation


def test_builds_honest_machine_specific_recommendation() -> None:
    recommendation = build_recommendation(*evidence(), safety_reserve_mib=1_024)

    assert recommendation["verdict"] == "CONDITIONAL"
    assert recommendation["live_cache_enabled"] is False
    assert recommendation["replay"]["policy"] == "static_hotset"
    assert recommendation["replay"]["capacity_per_layer"] == 8
    assert recommendation["hardware"]["remaining_configurable_headroom_mib"] == 6_976
    assert recommendation["provenance"] == {
        "hardware": "measured",
        "baseline": "measured",
        "locality": "measured",
        "policy": "estimated",
    }
    assert "EXPERT_BYTES_NOT_MEASURED" in recommendation["reason_codes"]
    assert "TRANSFER_TIMING_NOT_MEASURED" in recommendation["reason_codes"]


def test_rejects_simulation_mislabeled_as_measured() -> None:
    doctor, baseline, profile, simulation = evidence()
    simulation["measurement_kind"] = "measured"

    with pytest.raises(RecommendationInputError, match="simulation.*estimated"):
        build_recommendation(doctor, baseline, profile, simulation)
