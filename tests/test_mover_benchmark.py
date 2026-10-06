import pytest

import expertflow.runtime.cuda_transfer as cuda_transfer
from expertflow.runtime.cuda_transfer import benchmark_mover


def _fake_runtime(measure_calls: list[str]):
    class FakeCudaRuntime:
        def __init__(self, library_path, *, device):
            assert device == 0

        def versions(self):
            return {"runtime": 12080, "driver": 13010}

        def measure_contiguous_batch(
            self, slot_bytes, slot_count, *, source_memory, batches, warmup_copies
        ):
            measure_calls.append(("contiguous", slot_bytes, slot_count))
            return {"batched": [0.5, 0.7], "individual": [1.0, 1.4]}

        def measure_queue_depth(
            self, slot_bytes, depth, *, source_memory, batches, warmup_copies
        ):
            measure_calls.append(("queue", slot_bytes, depth))
            return [0.2, 0.3], [0.6]

        def measure_ready_latency(
            self,
            slot_bytes,
            *,
            source_memory,
            background_bytes,
            background_copies,
            samples,
            warmup_copies,
        ):
            measure_calls.append(
                ("ready", slot_bytes, background_bytes, background_copies)
            )
            if background_bytes == 0:
                return [0.05, 0.06]
            return [4.0, 6.0]

    return FakeCudaRuntime


def test_mover_report_contains_all_sections(tmp_path, monkeypatch) -> None:
    runtime = tmp_path / "cudart64_12.dll"
    runtime.write_bytes(b"test-runtime")
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(cuda_transfer, "CudaRuntime", _fake_runtime(calls))

    report = benchmark_mover(
        runtime,
        slot_bytes_values=(1024,),
        slot_counts=(8,),
        queue_depths=(1, 4),
        batches=2,
        warmup_copies=1,
        ready_samples=2,
        background_bytes=268435456,
        background_copies=4,
    )

    contract = report["contract"]
    assert report["measurement_kind"] == "measured"
    assert contract["slot_bytes_values"] == [1024]
    assert contract["staging_mode"] == "pinned"

    contiguous = report["contiguous_batch"]
    assert len(contiguous) == 1
    run = contiguous[0]
    assert run["payload_bytes"] == 8192
    assert run["batched"]["sample_count"] == 2
    assert run["individual"]["p50_ms"] == pytest.approx(1.2)
    assert run["speedup_ratio"] == pytest.approx(2.0)

    queue = report["queue_depth"]
    assert [entry["queue_depth"] for entry in queue] == [1, 4]
    assert queue[0]["per_copy_event"]["sample_count"] == 2

    ready = report["ready_latency"]
    assert [entry["load"] for entry in ready] == ["idle", "copy_engine_busy"]
    assert ready[1]["host_ready_latency"]["p50_ms"] == pytest.approx(5.0)

    assert ("contiguous", 1024, 8) in calls
    assert ("queue", 1024, 4) in calls
    assert ("ready", 1024, 0, 0) in calls
    assert ("ready", 1024, 268435456, 4) in calls


def test_mover_skips_loaded_ready_leg_without_background(
    tmp_path, monkeypatch
) -> None:
    runtime = tmp_path / "cudart64_12.dll"
    runtime.write_bytes(b"runtime")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(cuda_transfer, "CudaRuntime", _fake_runtime(calls))
    report = benchmark_mover(
        runtime,
        slot_bytes_values=(512,),
        slot_counts=(4,),
        queue_depths=(2,),
        batches=1,
        warmup_copies=0,
        ready_samples=1,
        background_bytes=0,
        background_copies=0,
    )

    assert [entry["load"] for entry in report["ready_latency"]] == ["idle"]
    assert not any(call[0] == "ready" and call[-1] != 0 for call in calls)


@pytest.mark.parametrize(
    "overrides",
    [
        {"slot_bytes_values": ()},
        {"slot_counts": (0,)},
        {"queue_depths": ()},
        {"batches": 0},
        {"warmup_copies": -1},
        {"ready_samples": 0},
        {"background_bytes": -1},
        {"device": -1},
    ],
)
def test_mover_rejects_invalid_contracts(tmp_path, overrides) -> None:
    runtime = tmp_path / "cudart64_12.dll"
    runtime.write_bytes(b"runtime")
    kwargs = {
        "slot_bytes_values": (1024,),
        "slot_counts": (8,),
        "queue_depths": (2,),
        "batches": 1,
        "warmup_copies": 0,
        "ready_samples": 1,
        "background_bytes": 0,
        "background_copies": 0,
        "staging_mode": "pinned",
        "device": 0,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        benchmark_mover(runtime, **kwargs)


def test_mover_rejects_unknown_staging_mode(tmp_path) -> None:
    runtime = tmp_path / "cudart64_12.dll"
    runtime.write_bytes(b"runtime")
    with pytest.raises(ValueError):
        benchmark_mover(
            runtime,
            slot_bytes_values=(1024,),
            slot_counts=(8,),
            queue_depths=(2,),
            batches=1,
            warmup_copies=0,
            ready_samples=1,
            background_bytes=0,
            background_copies=0,
            staging_mode="unified",
        )


def test_mover_rejects_partial_background_contract(tmp_path) -> None:
    runtime = tmp_path / "cudart64_12.dll"
    runtime.write_bytes(b"runtime")
    with pytest.raises(ValueError):
        benchmark_mover(
            runtime,
            slot_bytes_values=(1024,),
            slot_counts=(8,),
            queue_depths=(2,),
            batches=1,
            warmup_copies=0,
            ready_samples=1,
            background_bytes=4096,
            background_copies=0,
        )
