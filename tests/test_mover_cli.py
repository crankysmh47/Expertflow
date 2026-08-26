import json
from pathlib import Path

import expertflow.cli.main as cli


def test_mover_benchmark_cli_passes_full_contract(
    tmp_path: Path, monkeypatch
) -> None:
    cudart = tmp_path / "cudart64_12.dll"
    output = tmp_path / "mover.json"
    cudart.write_bytes(b"runtime")
    captured: dict[str, object] = {}

    def fake_benchmark(runtime, **kwargs):
        captured.update(kwargs)
        captured["runtime"] = runtime
        return {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "contiguous_batch": [],
            "queue_depth": [],
            "ready_latency": [],
        }

    monkeypatch.setattr(cli, "benchmark_mover", fake_benchmark)

    result = cli.main(
        [
            "mover-benchmark",
            "--cudart",
            str(cudart),
            "--slot-bytes",
            "3346048",
            "--slot-bytes",
            "26768384",
            "--slot-count",
            "1",
            "--slot-count",
            "8",
            "--queue-depth",
            "2",
            "--queue-depth",
            "8",
            "--batches",
            "12",
            "--warmup-copies",
            "3",
            "--ready-samples",
            "40",
            "--background-bytes",
            "67108864",
            "--background-copies",
            "4",
            "--staging-mode",
            "pinned_wc",
            "--device",
            "0",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert captured == {
        "runtime": cudart.resolve(),
        "slot_bytes_values": (3346048, 26768384),
        "slot_counts": (1, 8),
        "queue_depths": (2, 8),
        "batches": 12,
        "warmup_copies": 3,
        "ready_samples": 40,
        "background_bytes": 67108864,
        "background_copies": 4,
        "staging_mode": "pinned_wc",
        "device": 0,
    }
    assert json.loads(output.read_text(encoding="utf-8"))[
        "measurement_kind"
    ] == "measured"
