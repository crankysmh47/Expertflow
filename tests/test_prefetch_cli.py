import json
from pathlib import Path

from expertflow.cli.main import main
from expertflow.trace.schema import SCHEMA_VERSION


def _trace_line(
    conversation_id: str,
    forward_id: int,
    layer_id: int,
    experts: list[int],
) -> str:
    record = {
        "schema_version": SCHEMA_VERSION,
        "request_id": f"{conversation_id}-r",
        "conversation_id": conversation_id,
        "turn_index": 0,
        "phase": "decode",
        "forward_id": forward_id,
        "hook_order": 0,
        "token_index": forward_id,
        "token_id": 11,
        "layer_id": layer_id,
        "selected_expert_ids": experts,
        "selected_expert_weights": None,
        "observed_at_ns": 0,
    }
    return json.dumps(record)


def test_prefetch_sim_cli_writes_estimated_report(
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "conv-a.jsonl"
    trace_b = tmp_path / "conv-b.jsonl"
    output = tmp_path / "prefetch.json"
    trace_a.write_text(
        "\n".join(
            [
                _trace_line("a", 0, 0, [1, 2]),
                _trace_line("a", 1, 0, [3, 4]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_b.write_text(
        "\n".join(
            [
                _trace_line("b", 0, 0, [9]),
                _trace_line("b", 1, 0, [3]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = main(
        [
            "prefetch-sim",
            str(trace_a),
            str(trace_b),
            "--prediction",
            "oracle",
            "--capacity-per-layer",
            "8",
            "--max-transfers-per-step",
            "8",
            "--expert-transfer-ms",
            "0.273",
            "--slot-bytes",
            "3346048",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["measurement_kind"] == "estimated"
    assert report["source_traces"] == [
        str(trace_a.resolve()),
        str(trace_b.resolve()),
    ]
    simulation = report["simulation"]
    assert simulation["prediction"] == "oracle"
    assert simulation["conversation_count"] == 2
    assert simulation["step_count"] == 2
    assert simulation["demand_count"] == 6
    assert simulation["miss_count"] == 0


def test_prefetch_sim_cli_rejects_duplicate_conversations(
    tmp_path: Path,
) -> None:
    trace_one = tmp_path / "one.jsonl"
    trace_two = tmp_path / "two.jsonl"
    trace_one.write_text(_trace_line("a", 0, 0, [1]) + "\n", encoding="utf-8")
    trace_two.write_text(_trace_line("a", 0, 0, [2]) + "\n", encoding="utf-8")

    result = main(
        [
            "prefetch-sim",
            str(trace_one),
            str(trace_two),
            "--capacity-per-layer",
            "4",
            "--expert-transfer-ms",
            "0.2",
            "--slot-bytes",
            "1024",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    assert result != 0


def test_prefetch_sim_cli_rejects_mixed_conversation_trace(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "mixed.jsonl"
    trace.write_text(
        "\n".join([_trace_line("a", 0, 0, [1]), _trace_line("b", 1, 0, [2])])
        + "\n",
        encoding="utf-8",
    )

    result = main(
        [
            "prefetch-sim",
            str(trace),
            "--capacity-per-layer",
            "4",
            "--expert-transfer-ms",
            "0.2",
            "--slot-bytes",
            "1024",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    assert result != 0
