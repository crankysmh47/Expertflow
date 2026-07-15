import json
from pathlib import Path

from expertflow.cli.main import main


def test_replay_cli_writes_standalone_html(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    recommendation = tmp_path / "recommendation.json"
    output = tmp_path / "report.html"
    trace.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "request_id": "req-001",
                "conversation_id": "conv-001",
                "turn_index": 0,
                "phase": "decode",
                "forward_id": 0,
                "hook_order": 0,
                "token_index": 0,
                "token_id": 42,
                "layer_id": 0,
                "selected_expert_ids": [1, 2],
                "selected_expert_weights": None,
                "observed_at_ns": 1_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recommendation.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "verdict": "CONDITIONAL",
                "live_cache_enabled": False,
                "hardware": {
                    "gpu_name": "Test GPU",
                    "total_vram_mib": 16_000,
                    "measured_peak_vram_mib": 8_000,
                    "safety_reserve_mib": 1_024,
                    "remaining_configurable_headroom_mib": 6_976,
                },
                "replay": {
                    "policy": "static_hotset",
                    "capacity_per_layer": 2,
                    "estimated_hit_rate": 1.0,
                    "estimated_lru_hit_rate": 0.0,
                },
                "reason_codes": ["STRATIFIED_TRACE_REQUIRED"],
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "replay",
            str(trace),
            "--recommendation",
            str(recommendation),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert "CONDITIONAL" in output.read_text(encoding="utf-8")


def test_replay_cli_combines_and_filters_multiple_traces(
    tmp_path: Path,
) -> None:
    traces = [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]
    training_trace = tmp_path / "training.jsonl"
    for trace in [*traces, training_trace]:
        records = []
        for phase, layer in (("prefill", 0), ("prefill", 1), ("decode", 0)):
            records.append(
                {
                    "schema_version": "1.0.0",
                    "request_id": "req-001",
                    "conversation_id": "conv-001",
                    "turn_index": 0,
                    "phase": phase,
                    "forward_id": 0,
                    "hook_order": len(records),
                    "token_index": 0,
                    "token_id": 42,
                    "layer_id": layer,
                    "selected_expert_ids": [1, 2],
                    "selected_expert_weights": None,
                    "observed_at_ns": 1_000 + len(records),
                }
            )
        trace.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    recommendation = tmp_path / "recommendation.json"
    output = tmp_path / "report.html"
    recommendation.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "verdict": "CONDITIONAL",
                "live_cache_enabled": False,
                "hardware": {
                    "gpu_name": "Test GPU",
                    "total_vram_mib": 16_000,
                    "measured_peak_vram_mib": 8_000,
                    "safety_reserve_mib": 1_024,
                    "remaining_configurable_headroom_mib": 6_976,
                },
                "replay": {
                    "policy": "static_hotset",
                    "capacity_per_layer": 2,
                    "estimated_hit_rate": 1.0,
                    "estimated_lru_hit_rate": 0.0,
                },
                "reason_codes": ["HELD_OUT_POLICY_REQUIRED"],
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "replay",
            str(traces[0]),
            str(traces[1]),
            "--phase",
            "prefill",
            "--max-layer",
            "0",
            "--fit-trace",
            str(training_trace),
            "--fit-phase",
            "decode",
            "--recommendation",
            str(recommendation),
            "--output",
            str(output),
        ]
    )

    html = output.read_text(encoding="utf-8")
    assert result == 0
    assert "2 token/layer events" in html
    assert "first.jsonl" in html
    assert "second.jsonl" in html
    assert "Training traces:" in html
    assert "training.jsonl" in html
    assert "--fit-phase decode" in html


def test_replay_cli_embeds_physical_evidence(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "request_id": "req-001",
                "conversation_id": "conv-001",
                "turn_index": 0,
                "phase": "decode",
                "forward_id": 0,
                "hook_order": 0,
                "token_index": 0,
                "token_id": 42,
                "layer_id": 0,
                "selected_expert_ids": [1, 2],
                "selected_expert_weights": None,
                "observed_at_ns": 1_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recommendation = tmp_path / "recommendation.json"
    recommendation.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "verdict": "CONDITIONAL",
                "live_cache_enabled": False,
                "hardware": {
                    "gpu_name": "Test GPU",
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
                "reason_codes": ["END_TO_END_CACHE_NOT_MEASURED"],
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        "heldout": {
            "capacity_per_layer": 96,
            "slot_bytes": 3_346_048,
            "layer_ids": [0],
            "lru_reset_scope": "conversation",
            "training_event_count": 1,
            "evaluation_event_count": 1,
            "excluded_shards": [],
            "aggregate": {
                "conversation_count": 1,
                "token_count": 1,
                "expert_demand_count": 2,
                "static_hotset": {"hit_rate": 0.5, "miss_count": 1, "cold_bytes": 3_346_048},
                "lru": {"hit_rate": 0.0, "miss_count": 2, "cold_bytes": 6_692_096},
            },
            "per_domain": [],
            "per_prompt": [],
        },
        "layout": {
            "measurement_kind": "measured_encoded_projected_cuda_layout",
            "object_count": 1,
            "encoded_object_bytes_min": 3_345_412,
            "encoded_object_bytes_max": 3_345_412,
            "projected_slot_bytes_min": 3_346_048,
            "projected_slot_bytes_max": 3_346_048,
            "alignment_bytes": 128,
            "projected_cache_bytes": 321_220_608,
            "projection": {"capacity_per_layer": 96, "slot_count": 96, "target_layer_count": 1},
        },
        "transfer": {
            "measurement_kind": "measured",
            "trial_count": 1,
            "runs": [
                {
                    "direction": "host_to_device",
                    "source_memory": "pinned",
                    "payload_bytes": 3_346_048,
                    "cuda_event_per_copy": {"mean_gib_per_second": 10.0},
                    "single_copy_cuda_event": {"p50_ms": 0.2, "p95_ms": 0.3, "sample_count": 10},
                    "host_enqueue": {"p50_ms": 0.001, "p95_ms": 0.002},
                },
                {
                    "direction": "host_to_device",
                    "source_memory": "pageable",
                    "payload_bytes": 3_346_048,
                    "single_copy_cuda_event": {"p50_ms": 0.3, "p95_ms": 0.4},
                },
            ],
        },
        "deadline": {
            "measurement_kind": "estimated_cross_backend",
            "expert_transfer_ms": 0.3,
            "blocking_no_prefetch_ms_per_token": 1.0,
            "observed_adjacent_layer_window_ms": {"median": 1.0, "p95": 2.0},
            "timing_evidence": {
                "contention_measured": False,
                "live_runtime_measurement": False,
                "transfer_backend": "cuda",
                "window_backend": "vulkan",
            },
            "one_layer_oracle": {"residual_blocking_ms_per_token": 0.1, "late_event_count": 1},
        },
    }
    paths = {}
    for name, payload in artifacts.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "report.html"

    result = main(
        [
            "replay",
            str(trace),
            "--recommendation",
            str(recommendation),
            "--heldout-breakdown",
            str(paths["heldout"]),
            "--expert-layout",
            str(paths["layout"]),
            "--transfer-evidence",
            str(paths["transfer"]),
            "--deadline-evidence",
            str(paths["deadline"]),
            "--output",
            str(output),
        ]
    )

    html = output.read_text(encoding="utf-8")
    assert result == 0
    assert "Static-96 contract" in html
    assert "--heldout-breakdown" in html
    assert "live_cache_enabled=false" in html
