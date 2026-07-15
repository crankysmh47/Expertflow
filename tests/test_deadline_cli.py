import json
from pathlib import Path

from expertflow.cli.main import main


def write_trace(
    path: Path, *, phase: str, experts: list[int], timestamps: tuple[int, int]
) -> None:
    records = []
    for layer_id, observed_at_ns in enumerate(timestamps):
        records.append(
            {
                "schema_version": "1.0.0",
                "request_id": "req-001",
                "conversation_id": "conv-001",
                "turn_index": 0,
                "phase": phase,
                "forward_id": 0,
                "hook_order": layer_id,
                "token_index": 0,
                "token_id": 100,
                "layer_id": layer_id,
                "selected_expert_ids": experts,
                "selected_expert_weights": None,
                "observed_at_ns": observed_at_ns,
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_deadline_cli_writes_backend_specific_oracle_artifact(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    output = tmp_path / "deadline.json"
    write_trace(
        training, phase="prefill", experts=[1, 2], timestamps=(0, 1)
    )
    write_trace(
        evaluation,
        phase="decode",
        experts=[3, 4],
        timestamps=(1_000_000, 3_000_000),
    )

    result = main(
        [
            "deadline-eval",
            "--train",
            str(training),
            "--eval",
            str(evaluation),
            "--train-phase",
            "prefill",
            "--eval-phase",
            "decode",
            "--max-layer",
            "1",
            "--capacity",
            "2",
            "--expert-transfer-ms",
            "0.5",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert report["token_count"] == 1
    assert report["one_layer_oracle"]["late_event_count"] == 1
    assert report["training_source_traces"] == [str(training.resolve())]
    assert report["evaluation_source_traces"] == [str(evaluation.resolve())]


def test_deadline_cli_loads_split_manifest_and_labels_cross_backend_estimate(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    manifest = tmp_path / "collection-manifest.json"
    output = tmp_path / "deadline.json"
    write_trace(
        training, phase="prefill", experts=[1, 2], timestamps=(0, 1)
    )
    write_trace(
        evaluation,
        phase="decode",
        experts=[3, 4],
        timestamps=(1_000_000, 3_000_000),
    )
    manifest.write_text(
        json.dumps(
            {
                "summary": {
                    "conversation_count": 2,
                    "passed": 2,
                    "failed": 0,
                },
                "shards": [
                    {
                        "conversation_id": "train-1",
                        "split": "train",
                        "domain": "code",
                        "latest_status": "passed",
                        "attempts": [
                            {"trace": {"artifact": {"path": str(training)}}}
                        ],
                    },
                    {
                        "conversation_id": "test-1",
                        "split": "test",
                        "domain": "math_reasoning",
                        "latest_status": "passed",
                        "attempts": [
                            {"trace": {"artifact": {"path": str(evaluation)}}}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "deadline-eval",
            "--collection-manifest",
            str(manifest),
            "--train-phase",
            "prefill",
            "--eval-phase",
            "decode",
            "--max-layer",
            "1",
            "--capacity",
            "2",
            "--expert-transfer-ms",
            "0.5",
            "--transfer-backend",
            "cuda_idle_microbenchmark",
            "--window-backend",
            "vulkan_router_callback",
            "--transfer-statistic",
            "pooled_p95",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert report["measurement_kind"] == "estimated_cross_backend"
    assert report["collection_manifest"] == str(manifest.resolve())
    assert report["evaluation_metadata"] == [
        {
            "conversation_id": "test-1",
            "domain": "math_reasoning",
            "source_trace": str(evaluation.resolve()),
            "split": "test",
        }
    ]
    assert report["excluded_shards"] == []
