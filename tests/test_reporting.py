from pathlib import Path

from expertflow.analysis.replay import replay_policy
from expertflow.reporting import render_replay_report
from expertflow.trace.schema import RouterTraceEvent


def event() -> RouterTraceEvent:
    return RouterTraceEvent(
        schema_version="1.0.0",
        request_id="req-001",
        conversation_id="conv-001",
        turn_index=0,
        phase="decode",
        forward_id=1,
        hook_order=1,
        token_index=2,
        token_id=42,
        layer_id=7,
        selected_expert_ids=(1, 2),
        selected_expert_weights=None,
        observed_at_ns=1_000,
    )


def recommendation() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "verdict": "CONDITIONAL",
        "live_cache_enabled": False,
        "hardware": {
            "gpu_name": "GPU <unsafe>",
            "total_vram_mib": 16_000,
            "measured_peak_vram_mib": 8_000,
            "safety_reserve_mib": 1_024,
            "remaining_configurable_headroom_mib": 6_976,
        },
        "replay": {
            "policy": "static_hotset",
            "capacity_per_layer": 2,
            "estimated_hit_rate": 0.5,
            "estimated_lru_hit_rate": 0.4,
        },
        "reason_codes": ["TRANSFER_TIMING_NOT_MEASURED"],
        "expert_cache": {
            "fit_scope": "in_sample",
            "projected_cache_mib": 6433.13671875,
            "remaining_headroom_after_cache_mib": 800.86328125,
            "measured_expert_transfer_ms": 0.2350416,
            "estimated_serial_h2d_ms_per_layer_sweep": 0.168936,
        },
    }


def test_renders_self_contained_escaped_replay_report() -> None:
    replay = replay_policy(
        (event(),), policy="static_hotset", capacity_per_layer=2
    )

    html = render_replay_report(
        recommendation(),
        replay,
        source_trace=Path("trace.jsonl"),
        recommendation_source=Path("recommendation.json"),
        reproduction_command="expertflow replay trace.jsonl",
    )

    assert "<!doctype html>" in html.lower()
    assert "GPU &lt;unsafe&gt;" in html
    assert "GPU <unsafe>" not in html
    assert "CONDITIONAL" in html
    assert "6,976 MiB" in html
    assert "6,433.14 MiB" in html
    assert "800.86 MiB" in html
    assert "IN-SAMPLE" in html
    assert "0.2350 ms" in html
    assert "MEASURED" in html
    assert "ESTIMATED" in html
    assert "TRANSFER_TIMING_NOT_MEASURED" in html
    assert "Trace schema" in html
    assert "Recommendation schema" in html
    assert "recommendation.json" in html
    assert "expertflow replay trace.jsonl" in html
    assert "Â" not in html
    assert "<script" not in html.lower()
    assert "src=\"http" not in html.lower()
    assert "href=\"http" not in html.lower()
