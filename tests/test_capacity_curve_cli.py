import json
from pathlib import Path

from expertflow.cli.main import main


def write_trace(path: Path) -> None:
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
                "token_id": 100,
                "layer_id": layer,
                "selected_expert_ids": [1, 2],
                "selected_expert_weights": None,
                "observed_at_ns": 1_000 + len(records),
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_capacity_curve_cli_filters_phase_and_layers(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "curve.json"
    write_trace(first)
    write_trace(second)

    result = main(
        [
            "capacity-curve",
            str(first),
            str(second),
            "--phase",
            "prefill",
            "--max-layer",
            "0",
            "--capacity",
            "2",
            "--slot-bytes",
            "100",
            "--expert-transfer-ms",
            "0.5",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert report["event_count"] == 2
    assert report["selection"] == {"phase": "prefill", "max_layer": 0}
    assert report["source_traces"] == [
        str(first.resolve()),
        str(second.resolve()),
    ]


def test_heldout_curve_cli_keeps_train_and_evaluation_sources_separate(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    output = tmp_path / "heldout.json"
    write_trace(training)
    write_trace(evaluation)

    result = main(
        [
            "heldout-curve",
            "--train",
            str(training),
            "--eval",
            str(evaluation),
            "--phase",
            "prefill",
            "--max-layer",
            "0",
            "--capacity",
            "2",
            "--slot-bytes",
            "100",
            "--expert-transfer-ms",
            "0.5",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert report["fit_scope"] == "held_out"
    assert report["training_source_traces"] == [str(training.resolve())]
    assert report["evaluation_source_traces"] == [str(evaluation.resolve())]
