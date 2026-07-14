import json
from pathlib import Path

from expertflow.cli.main import main


def trace_record(token_index: int, experts: list[int]) -> dict[str, object]:
    return {
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


def write_trace(path: Path) -> None:
    records = [trace_record(0, [2, 4]), trace_record(1, [2, 5])]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_profile_cli_writes_measured_report(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "profile.json"
    write_trace(trace_path)

    result = main(
        [
            "profile",
            str(trace_path),
            "--output",
            str(output_path),
            "--static-budget",
            "1",
            "--static-budget",
            "2",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["measurement_kind"] == "measured"
    assert report["source_trace"] == str(trace_path.resolve())
    assert report["profile"]["total_events"] == 2
    assert report["profile"]["layers"][0]["static_hit_rates"] == [
        [1, 0.5],
        [2, 0.75],
    ]
