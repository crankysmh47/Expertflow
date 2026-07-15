import json
from pathlib import Path

import expertflow.cli.main as cli


def test_transfer_benchmark_cli_writes_measured_report(
    tmp_path: Path, monkeypatch
) -> None:
    cudart = tmp_path / "cudart64_12.dll"
    output = tmp_path / "transfer.json"
    cudart.write_bytes(b"runtime")
    captured: dict[str, object] = {}

    def fake_benchmark(
        runtime,
        *,
        payload_bytes,
        batches,
        copies_per_batch,
        warmup_copies,
        single_copy_samples,
        device,
    ):
        captured.update(
            runtime=runtime,
            payload_bytes=payload_bytes,
            batches=batches,
            copies_per_batch=copies_per_batch,
            warmup_copies=warmup_copies,
            single_copy_samples=single_copy_samples,
            device=device,
        )
        return {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "runs": [],
        }

    monkeypatch.setattr(cli, "benchmark_cuda_transfers", fake_benchmark)

    result = cli.main(
        [
            "transfer-benchmark",
            "--cudart",
            str(cudart),
            "--payload-bytes",
            "3345412",
            "--payload-bytes",
            "26763296",
            "--batches",
            "12",
            "--copies-per-batch",
            "25",
            "--warmup-copies",
            "3",
            "--single-copy-samples",
            "40",
            "--device",
            "0",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert captured == {
        "runtime": cudart.resolve(),
        "payload_bytes": (3345412, 26763296),
        "batches": 12,
        "copies_per_batch": 25,
        "warmup_copies": 3,
        "single_copy_samples": 40,
        "device": 0,
    }
    assert json.loads(output.read_text(encoding="utf-8"))[
        "measurement_kind"
    ] == "measured"


def test_transfer_aggregate_cli_pools_declared_trials(
    tmp_path: Path, monkeypatch
) -> None:
    trial_one = tmp_path / "trial-01.json"
    trial_two = tmp_path / "trial-02.json"
    trial_one.write_text('{"trial":1}', encoding="utf-8")
    trial_two.write_text('{"trial":2}', encoding="utf-8")
    output = tmp_path / "aggregate.json"
    captured: dict[str, object] = {}

    def fake_aggregate(reports, *, source_paths):
        captured["reports"] = reports
        captured["source_paths"] = source_paths
        return {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "trial_count": len(reports),
        }

    monkeypatch.setattr(cli, "aggregate_cuda_transfer_trials", fake_aggregate)

    result = cli.main(
        [
            "transfer-aggregate",
            str(trial_one),
            str(trial_two),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert captured["reports"] == [{"trial": 1}, {"trial": 2}]
    assert captured["source_paths"] == (
        str(trial_one.resolve()),
        str(trial_two.resolve()),
    )
    assert json.loads(output.read_text(encoding="utf-8"))["trial_count"] == 2
