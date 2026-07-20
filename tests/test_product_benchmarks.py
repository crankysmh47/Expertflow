import pytest

from scripts.benchmark_product_server import aggregate_runs, choose_profile
from scripts.measure_context_frontier import choose_context_profile


def _run(mode: str, slots: int, rep: int, tps: float, peak: float = 11000.0, errors: int = 0):
    return {
        "mode": mode,
        "parallel_slots": slots,
        "repetition": rep,
        "aggregate_generated_tps": tps,
        "per_request_tps": [tps / slots] * slots,
        "request_latency_seconds": [2.0] * slots,
        "ttft_seconds": [0.4] * slots,
        "peak_process_owned_vram_mib": peak,
        "completed_requests": slots - errors,
        "errors": errors,
        "wall_seconds": 2.0,
        "generated_tokens": 128 * slots,
    }


def test_aggregate_runs_reports_mean_variance_and_latency() -> None:
    summary = aggregate_runs([_run("expertflow", 2, 1, 30.0), _run("expertflow", 2, 2, 32.0), _run("expertflow", 2, 3, 31.0)])
    row = summary[0]
    assert row["aggregate_generated_tps_mean"] == pytest.approx(31.0)
    assert row["aggregate_generated_tps_sample_variance"] == pytest.approx(1.0)
    assert row["median_request_latency_seconds"] == 2.0
    assert row["p95_request_latency_seconds"] == 2.0
    assert row["error_count"] == 0


def test_choose_profile_requires_stability_margin_and_improvement() -> None:
    rows = [
        {"mode": "expertflow", "parallel_slots": 1, "aggregate_generated_tps_mean": 28.0, "peak_process_owned_vram_mib": 11000.0, "error_count": 0, "p95_request_latency_seconds": 5.0},
        {"mode": "expertflow", "parallel_slots": 2, "aggregate_generated_tps_mean": 45.0, "peak_process_owned_vram_mib": 15000.0, "error_count": 0, "p95_request_latency_seconds": 6.0},
        {"mode": "expertflow", "parallel_slots": 4, "aggregate_generated_tps_mean": 47.0, "peak_process_owned_vram_mib": 15900.0, "error_count": 0, "p95_request_latency_seconds": 7.0},
    ]
    selected = choose_profile(rows, total_vram_mib=16311, margin_mib=512)
    assert selected["parallel_slots"] == 2


def test_context_profile_uses_measured_processed_tokens_and_margin() -> None:
    rows = [
        {"mode": "expertflow", "allocated_context": 8192, "processed_tokens": 8000, "status": "pass", "peak_process_owned_vram_mib": 12000.0},
        {"mode": "expertflow", "allocated_context": 16384, "processed_tokens": 16000, "status": "pass", "peak_process_owned_vram_mib": 15600.0},
        {"mode": "expertflow", "allocated_context": 32768, "processed_tokens": 0, "status": "oom", "peak_process_owned_vram_mib": 16000.0},
    ]
    selected = choose_context_profile(rows, total_vram_mib=16311, margin_mib=512)
    assert selected["allocated_context"] == 16384
    assert selected["processed_tokens"] == 16000
