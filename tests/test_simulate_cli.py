import json
from pathlib import Path

from expertflow.cli.main import main


def write_trace(path: Path) -> None:
    records = []
    for token_index, experts in enumerate(([1, 2], [1, 3], [1, 2])):
        records.append(
            {
                "schema_version": "1.0.0",
                "request_id": "req-001",
                "conversation_id": "conv-001",
                "turn_index": 0,
                "phase": "decode",
                "forward_id": token_index,
                "hook_order": token_index,
                "token_index": token_index,
                "token_id": 100 + token_index,
                "layer_id": 7,
                "selected_expert_ids": experts,
                "selected_expert_weights": None,
                "observed_at_ns": 1_000 + token_index,
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_simulate_cli_writes_estimated_policy_report(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "simulation.json"
    write_trace(trace_path)

    result = main(
        [
            "simulate",
            str(trace_path),
            "--capacity-per-layer",
            "2",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["measurement_kind"] == "estimated"
    assert report["simulation"]["reactive"]["miss_count"] == 6
    assert report["simulation"]["static_hotset"]["hit_count"] == 5
    assert report["simulation"]["lru"]["hit_count"] == 2
