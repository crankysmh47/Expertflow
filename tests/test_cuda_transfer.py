import pytest

import expertflow.runtime.cuda_transfer as cuda_transfer
from expertflow.runtime.cuda_transfer import (
    aggregate_cuda_transfer_trials,
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
    assert summary["p50_ms"] == pytest.approx(2.5)
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

        def measure_single_copy(
            self, payload_bytes, *, source_memory, samples, warmup_copies
        ):
            assert source_memory in {"pageable", "pinned"}
            assert samples == 4
            assert warmup_copies == 1
            return [0.11, 0.12, 0.13, 0.14], [0.01, 0.02, 0.03, 0.04]

    monkeypatch.setattr(cuda_transfer, "CudaRuntime", FakeCudaRuntime)

    report = benchmark_cuda_transfers(
        runtime,
        payload_bytes=(1024,),
        batches=2,
        copies_per_batch=3,
        warmup_copies=1,
        single_copy_samples=4,
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
    assert runs[2]["single_copy_cuda_event"]["p50_ms"] == 0.125
    assert runs[2]["host_enqueue"]["p95_ms"] == 0.04
    assert report["contract"]["single_copy_samples"] == 4


def test_pools_raw_transfer_samples_across_independent_trials() -> None:
    def trial(first: float, second: float) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "measurement_kind": "measured",
            "runtime": {"sha256": "a" * 64},
            "device_index": 0,
            "contract": {"single_copy_samples": 2},
            "runs": [
                {
                    "payload_bytes": 1024,
                    "source_memory": "pinned",
                    "direction": "host_to_device",
                    "raw_cuda_event_ms_per_copy": [first, second],
                    "raw_host_wall_ms_per_copy": [first + 0.1, second + 0.1],
                    "raw_single_copy_cuda_event_ms": [first, second],
                    "raw_host_enqueue_ms": [first / 10, second / 10],
                }
            ],
        }

    report = aggregate_cuda_transfer_trials(
        [trial(1.0, 2.0), trial(3.0, 4.0)],
        source_paths=("trial-01.json", "trial-02.json"),
    )

    assert report["trial_count"] == 2
    assert report["source_trials"] == ["trial-01.json", "trial-02.json"]
    run = report["runs"][0]
    assert run["single_copy_cuda_event"]["sample_count"] == 4
    assert run["single_copy_cuda_event"]["p50_ms"] == 2.5
    assert run["single_copy_cuda_event"]["p95_ms"] == 4.0
    assert run["host_enqueue"]["p50_ms"] == 0.25
