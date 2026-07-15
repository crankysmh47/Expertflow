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


def physical_evidence() -> dict[str, object]:
    return {
        "heldout_breakdown": {
            "capacity_per_layer": 96,
            "slot_bytes": 3_346_048,
            "layer_ids": list(range(21)),
            "lru_reset_scope": "conversation",
            "training_event_count": 40_740,
            "evaluation_event_count": 10_584,
            "excluded_shards": [{"conversation_id": "bad <shard>"}],
            "aggregate": {
                "conversation_count": 8,
                "token_count": 504,
                "expert_demand_count": 84_672,
                "static_hotset": {
                    "hit_rate": 0.875720427,
                    "miss_count": 10_523,
                    "cold_bytes": 35_210_463_104,
                },
                "lru": {
                    "hit_rate": 0.863390495,
                    "miss_count": 11_567,
                    "cold_bytes": 38_703_737_216,
                },
            },
            "per_domain": [
                {
                    "domain": "code <unsafe>",
                    "static_hotset": {"hit_rate": 0.8581, "miss_count": 1_502},
                    "lru": {"hit_rate": 0.9005, "miss_count": 1_053},
                }
            ],
            "per_prompt": [
                {
                    "conversation_id": "validation-code-08",
                    "split": "validation",
                    "domain": "code",
                    "static_hotset": {"hit_rate": 0.8581, "miss_count": 1_502},
                    "lru": {"hit_rate": 0.9005, "miss_count": 1_053},
                }
            ],
        },
        "expert_layout": {
            "measurement_kind": "measured_encoded_projected_cuda_layout",
            "object_count": 3_840,
            "encoded_object_bytes_min": 3_345_412,
            "encoded_object_bytes_max": 3_345_412,
            "projected_slot_bytes_min": 3_346_048,
            "projected_slot_bytes_max": 3_346_048,
            "alignment_bytes": 128,
            "projected_cache_bytes": 6_745_632_768,
            "projection": {
                "capacity_per_layer": 96,
                "slot_count": 2_016,
                "target_layer_count": 21,
            },
        },
        "transfer": {
            "measurement_kind": "measured",
            "trial_count": 3,
            "runs": [
                {
                    "direction": "host_to_device",
                    "source_memory": "pinned",
                    "payload_bytes": 3_346_048,
                    "cuda_event_per_copy": {"mean_gib_per_second": 13.3503},
                    "single_copy_cuda_event": {
                        "p50_ms": 0.234016,
                        "p95_ms": 0.234272,
                        "sample_count": 600,
                    },
                    "host_enqueue": {"p50_ms": 0.0013, "p95_ms": 0.0039},
                },
                {
                    "direction": "host_to_device",
                    "source_memory": "pageable",
                    "payload_bytes": 3_346_048,
                    "single_copy_cuda_event": {
                        "p50_ms": 0.264672,
                        "p95_ms": 0.276704,
                    },
                },
            ],
        },
        "deadline": {
            "measurement_kind": "estimated_cross_backend",
            "expert_transfer_ms": 0.236288,
            "blocking_no_prefetch_ms_per_token": 4.93345,
            "observed_adjacent_layer_window_ms": {"median": 1.5307, "p95": 2.072},
            "timing_evidence": {
                "contention_measured": False,
                "live_runtime_measurement": False,
                "transfer_backend": "cuda_idle",
                "window_backend": "vulkan_callback",
            },
            "one_layer_oracle": {
                "residual_blocking_ms_per_token": 0.170541,
                "late_event_count": 212,
            },
        },
        "sources": {
            "heldout_breakdown": "breakdown <unsafe>.json",
            "expert_layout": "layout.json",
            "transfer": "transfer.json",
            "deadline": "deadline.json",
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
        physical_evidence=physical_evidence(),
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
    assert "Static-96 contract" in html
    assert "96 slots per target layer" in html
    assert "3,346,048 bytes" in html
    assert "6,433.14 MiB" in html
    assert "87.57%" in html
    assert "86.34%" in html
    assert "validation-code-08" in html
    assert "code &lt;unsafe&gt;" in html
    assert "code <unsafe>" not in html
    assert "0.2340 ms" in html
    assert "0.2343 ms" in html
    assert "ESTIMATED CROSS-BACKEND" in html
    assert "live_cache_enabled=false" in html
    assert "breakdown &lt;unsafe&gt;.json" in html
    assert "Â" not in html
    assert "<script" not in html.lower()
    assert "src=\"http" not in html.lower()
    assert "href=\"http" not in html.lower()
