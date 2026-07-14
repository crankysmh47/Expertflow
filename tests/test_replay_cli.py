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
