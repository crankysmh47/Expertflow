import json
from pathlib import Path

import pytest

from expertflow.predictor.temporal_sidecar_analysis import (
    analyze_prevented_misses,
    compare_t2_pair,
    summarize_t2_run,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def test_sidecar_summary_separates_useful_late_waste_and_blocking(
    tmp_path: Path,
) -> None:
    performance = tmp_path / "performance.json"
    cache = tmp_path / "cache.jsonl"
    sidecar = tmp_path / "t2.jsonl"
    _write_json(
        performance,
        {
            "prompt_tokens": 40,
            "generated_tokens": 10,
            "prompt_eval_ms": 2000.0,
            "decode_eval_ms": 500.0,
            "end_to_end_ms": 2500.0,
            "time_to_first_token_ms": 2001.0,
            "decode_token_latencies_ms": [40.0, 50.0, 60.0],
        },
    )
    _write_jsonl(
        cache,
        [
            {
                "selected": [1, 2, 3, 4, 5, 6, 7, 8],
                "physical_slots": [0, 1, 2, 3, 4, 5, 6, 32],
                "hits": 5,
                "blocking_misses": 2,
                "bytes_transferred": 6690824,
                "blocking_duration_us": 2200,
            }
        ],
    )
    _write_jsonl(
        sidecar,
        [
            {
                "record_kind": "prefetch",
                "transfer_enqueued": True,
                "outcome": "ready_useful",
                "bytes": 3345412,
                "blocking_wait_us": 0,
                "staging_ns": 100,
                "enqueue_ns": 20,
                "queue_to_ready_ns": 400,
                "h2d_cuda_event_ms": 0.25,
            },
            {
                "record_kind": "prefetch",
                "transfer_enqueued": True,
                "outcome": "late_useful",
                "bytes": 3345412,
                "blocking_wait_us": 300,
                "staging_ns": 110,
                "enqueue_ns": 25,
                "queue_to_ready_ns": 450,
                "h2d_cuda_event_ms": 0.30,
            },
            {
                "record_kind": "prefetch",
                "transfer_enqueued": True,
                "outcome": "wasted",
                "bytes": 3345412,
                "blocking_wait_us": 0,
                "staging_ns": 120,
                "enqueue_ns": 30,
                "queue_to_ready_ns": 500,
                "h2d_cuda_event_ms": 0.35,
            },
            {
                "record_kind": "summary",
                "records": 3,
                "enqueued": 3,
                "ready_useful": 1,
                "late_useful": 1,
                "wasted": 1,
                "arena_bytes": 113744640,
            },
        ],
    )

    summary = summarize_t2_run(performance, cache, sidecar)
    assert summary["ready_useful_prefetches"] == 1
    assert summary["late_useful_prefetches"] == 1
    assert summary["wasted_prefetches"] == 1
    assert summary["sidecar_demands"] == 1
    assert summary["expert_demands"] == 8
    assert summary["blocking_misses"] == 2
    assert summary["reactive_blocking_ms"] == 2.2
    assert summary["sidecar_blocking_ms"] == 0.3
    assert summary["wasted_bytes"] == 3345412
    assert summary["arena_bytes"] == 113744640
    assert summary["prompt_eval_tps"] == 20.0
    assert summary["decode_tps"] == 20.0
    assert summary["token_latency_p50_ms"] == 50.0
    assert summary["token_latency_p95_ms"] == pytest.approx(59.0)


def test_pair_comparison_reports_blocking_and_throughput_changes() -> None:
    comparison = compare_t2_pair(
        {
            "decode_tps": 10.0,
            "prompt_eval_tps": 20.0,
            "end_to_end_ms": 3000.0,
            "blocking_misses": 12,
            "total_blocking_ms": 30.0,
        },
        {
            "decode_tps": 11.0,
            "prompt_eval_tps": 19.0,
            "end_to_end_ms": 2900.0,
            "blocking_misses": 9,
            "total_blocking_ms": 20.0,
        },
    )
    assert comparison["decode_tps_percent"] == pytest.approx(10.0)
    assert comparison["prompt_eval_tps_percent"] == pytest.approx(-5.0)
    assert comparison["end_to_end_percent"] == pytest.approx(3.3333333333)
    assert comparison["blocking_miss_reduction_percent"] == 25.0
    assert comparison["blocking_time_reduction_percent"] == pytest.approx(
        33.3333333333
    )


def test_prevented_miss_analysis_pairs_sidecar_demands_with_reactive_loads(
    tmp_path: Path,
) -> None:
    reactive = tmp_path / "reactive.jsonl"
    predictive = tmp_path / "predictive.jsonl"
    common = {
        "layer_id": 24,
        "selected": [10, 11, 12, 13, 14, 15, 16, 17],
    }
    _write_jsonl(
        reactive,
        [
            {
                **common,
                "token_index": 7,
                "physical_slots": list(range(8)),
                "loads": [
                    {"expert": 10, "slot": 0},
                    {"expert": 11, "slot": 1},
                ],
            }
        ],
    )
    _write_jsonl(
        predictive,
        [
            {
                **common,
                "token_index": 7,
                "physical_slots": [32, 1, 2, 3, 4, 5, 6, 7],
                "loads": [{"expert": 11, "slot": 1}],
            }
        ],
    )

    result = analyze_prevented_misses(reactive, predictive)
    assert result == {
        "paired_sidecar_demands": 1,
        "actual_blocking_misses_prevented": 1,
        "sidecar_demands_that_were_reactive_hits": 0,
    }


def test_prevented_miss_analysis_rejects_nonidentical_routes(
    tmp_path: Path,
) -> None:
    reactive = tmp_path / "reactive.jsonl"
    predictive = tmp_path / "predictive.jsonl"
    _write_jsonl(
        reactive,
        [{"layer_id": 24, "token_index": 0, "selected": list(range(8))}],
    )
    _write_jsonl(
        predictive,
        [{"layer_id": 24, "token_index": 0, "selected": list(range(1, 9))}],
    )
    with pytest.raises(ValueError, match="router selections differ"):
        analyze_prevented_misses(reactive, predictive)


def test_cache_demand_accounting_rejects_unexplained_rank(tmp_path: Path) -> None:
    performance = tmp_path / "performance.json"
    cache = tmp_path / "cache.jsonl"
    sidecar = tmp_path / "t2.jsonl"
    _write_json(
        performance,
        {
            "prompt_tokens": 1,
            "generated_tokens": 1,
            "prompt_eval_ms": 1,
            "decode_eval_ms": 1,
            "end_to_end_ms": 2,
            "decode_token_latencies_ms": [],
        },
    )
    _write_jsonl(
        cache,
        [
            {
                "selected": list(range(8)),
                "physical_slots": list(range(8)),
                "hits": 3,
                "blocking_misses": 3,
                "bytes_transferred": 0,
                "blocking_duration_us": 0,
            }
        ],
    )
    _write_jsonl(
        sidecar,
        [{"record_kind": "summary", "records": 0, "enqueued": 0}],
    )
    with pytest.raises(ValueError, match="cache demands do not reconcile"):
        summarize_t2_run(performance, cache, sidecar)
