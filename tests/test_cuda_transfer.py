import pytest

import expertflow.runtime.cuda_transfer as cuda_transfer
from expertflow.runtime.cuda_transfer import (
    benchmark_cuda_transfers,
    summarize_latency_samples,
)


def test_summarizes_cuda_event_latency_and_throughput() -> None:
    summary = summarize_latency_samples(
        payload_bytes=4 * 1024 * 1024,
        samples_ms=[1.0, 2.0, 3.0, 4.0],
    )

    assert summary["sample_count"] == 4
    assert summary["min_ms"] == pytest.approx(1.0)
    assert summary["median_ms"] == pytest.approx(2.5)
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(4.0)
    assert summary["mean_gib_per_second"] == pytest.approx(1.5625)


@pytest.mark.parametrize(
    ("payload_bytes", "samples_ms"),
    [(0, [1.0]), (-1, [1.0]), (1, []), (1, [0.0]), (1, [-1.0])],
)
def test_rejects_invalid_transfer_samples(
    payload_bytes: int, samples_ms: list[float]
) -> None:
    with pytest.raises(ValueError):
        summarize_latency_samples(payload_bytes, samples_ms)


def test_builds_pageable_staging_and_host_to_device_curve(
    tmp_path, monkeypatch
) -> None:
    runtime = tmp_path / "cudart.dll"
    runtime.write_bytes(b"test-runtime")

    class FakeCudaRuntime:
        def __init__(self, library_path, *, device):
            assert library_path == runtime.resolve()
            assert device == 0

        def versions(self):
            return {"runtime": 12040, "driver": 13010}

        def measure_pageable_to_pinned(self, payload_bytes, **contract):
            assert contract == {
                "batches": 2,
                "copies_per_batch": 3,
                "warmup_copies": 1,
            }
            return [0.2, 0.3]

        def measure(self, payload_bytes, *, source_memory, **contract):
            assert source_memory in {"pageable", "pinned"}
            assert contract == {
                "batches": 2,
                "copies_per_batch": 3,
                "warmup_copies": 1,
            }
            return [0.1, 0.2], [0.15, 0.25]

    monkeypatch.setattr(cuda_transfer, "CudaRuntime", FakeCudaRuntime)

    report = benchmark_cuda_transfers(
        runtime,
        payload_bytes=(1024,),
        batches=2,
        copies_per_batch=3,
        warmup_copies=1,
        device=0,
    )

    runs = report["runs"]
    assert len(runs) == 3
    assert runs[0]["direction"] == "pageable_to_pinned"
    assert runs[0]["host_wall_per_copy"]["sample_count"] == 2
    assert [run["source_memory"] for run in runs[1:]] == [
        "pageable",
        "pinned",
    ]
