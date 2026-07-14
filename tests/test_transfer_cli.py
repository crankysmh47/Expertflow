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
        device,
    ):
        captured.update(
            runtime=runtime,
            payload_bytes=payload_bytes,
            batches=batches,
            copies_per_batch=copies_per_batch,
            warmup_copies=warmup_copies,
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
        "device": 0,
    }
    assert json.loads(output.read_text(encoding="utf-8"))[
        "measurement_kind"
    ] == "measured"
