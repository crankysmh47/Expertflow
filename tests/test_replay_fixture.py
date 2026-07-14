import json
from pathlib import Path

import pytest

from expertflow.analysis.cache_sim import simulate_policies
from expertflow.recommendation import build_recommendation
from expertflow.trace.io import load_router_events


FIXTURE_ROOT = Path(__file__).parents[1] / "examples" / "replay"


def test_checked_in_replay_fixture_reproduces_expected_totals() -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected.json").read_text(encoding="utf-8")
    )
    events = list(load_router_events(FIXTURE_ROOT / "trace.jsonl"))

    report = simulate_policies(
        events,
        capacity_per_layer=expected["capacity_per_layer"],
    )

    assert expected["source_measurement_kind"] == "previously_measured"
    assert report.measurement_kind == "estimated"
    assert len(events) == expected["event_count"]
    assert report.static_hotset.hit_count == expected["static_hotset"]["hits"]
    assert report.static_hotset.miss_count == expected["static_hotset"]["misses"]
    assert report.static_hotset.hit_rate == pytest.approx(
        expected["static_hotset"]["hit_rate"]
    )
    assert report.lru.hit_count == expected["lru"]["hits"]
    assert report.lru.miss_count == expected["lru"]["misses"]
    assert report.lru.hit_rate == pytest.approx(expected["lru"]["hit_rate"])

    machine = expected["machine_evidence"]
    recommendation = build_recommendation(
        {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "gpus": [
                {
                    "index": 0,
                    "name": machine["gpu_name"],
                    "memory_total_mib": machine["total_vram_mib"],
                }
            ],
        },
        {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "status": "passed",
            "memory": {
                "peak_gpu_used_mib": {
                    "0": machine["measured_peak_vram_mib"]
                }
            },
        },
        {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "profile": {"total_events": len(events), "layers": [{"layer_id": 0}]},
        },
        {
            "schema_version": "1.0.0",
            "measurement_kind": "estimated",
            "simulation": {
                "capacity_per_layer": expected["capacity_per_layer"],
                "static_hotset": {"hit_rate": report.static_hotset.hit_rate},
                "lru": {"hit_rate": report.lru.hit_rate},
            },
        },
        safety_reserve_mib=machine["safety_reserve_mib"],
    )
    expected_recommendation = expected["recommendation"]
    assert recommendation["verdict"] == expected_recommendation["verdict"]
    assert recommendation["live_cache_enabled"] is False
    assert recommendation["replay"]["policy"] == expected_recommendation["policy"]
    assert (
        recommendation["hardware"]["remaining_configurable_headroom_mib"]
        == expected_recommendation["remaining_configurable_headroom_mib"]
    )
