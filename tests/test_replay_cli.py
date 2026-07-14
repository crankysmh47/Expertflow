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
