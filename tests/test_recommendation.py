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


def test_supersedes_single_trace_policy_with_stratified_capacity_curve() -> None:
    curve = {
        "schema_version": "1.0.0",
        "measurement_kind": "estimated",
        "fit_scope": "in_sample",
        "event_count": 2_688,
        "expert_demand_count": 21_504,
        "layer_ids": [0, 1],
        "router_top_k": 8,
        "slot_bytes": 3_346_048,
        "expert_transfer_ms": 0.235,
        "points": [
            {
                "capacity_per_layer": 64,
                "projected_cache_bytes": 4_000_000_000,
                "static_hotset": {
                    "hit_rate": 0.92,
                    "estimated_serial_h2d_ms_per_layer_sweep": 2.8,
                },
                "lru": {
                    "hit_rate": 0.84,
                    "estimated_serial_h2d_ms_per_layer_sweep": 6.1,
                },
            },
            {
                "capacity_per_layer": 96,
                "projected_cache_bytes": 9_000_000_000,
                "static_hotset": {
                    "hit_rate": 0.99,
                    "estimated_serial_h2d_ms_per_layer_sweep": 0.2,
                },
                "lru": {
                    "hit_rate": 0.90,
                    "estimated_serial_h2d_ms_per_layer_sweep": 3.9,
                },
            },
        ],
    }

    recommendation = build_recommendation(
        *evidence(), capacity_curve=curve, safety_reserve_mib=1_024
    )

    assert recommendation["verdict"] == "CONDITIONAL"
    assert recommendation["live_cache_enabled"] is False
    assert recommendation["replay"]["capacity_per_layer"] == 64
    assert recommendation["replay"]["policy"] == "static_hotset"
    assert recommendation["expert_cache"]["projected_cache_bytes"] == 4_000_000_000
    assert recommendation["expert_cache"]["fit_scope"] == "in_sample"
    assert "EXPERT_BYTES_NOT_MEASURED" not in recommendation["reason_codes"]
    assert "TRANSFER_TIMING_NOT_MEASURED" not in recommendation["reason_codes"]
    assert "STRATIFIED_TRACE_REQUIRED" not in recommendation["reason_codes"]
    assert "HELD_OUT_POLICY_REQUIRED" in recommendation["reason_codes"]
    assert "PER_LAYER_DEADLINES_NOT_MEASURED" in recommendation["reason_codes"]
