import json

import pytest

from expertflow.benchmark.performance import (
    compare_modes,
    parse_cache_events,
    parse_probe_result,
    summarize_repetitions,
)


def test_parse_probe_result_calculates_rates_and_latency(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "prompt_tokens": 10,
                "generated_tokens": 4,
                "prompt_eval_ms": 200.0,
                "decode_eval_ms": 400.0,
                "end_to_end_ms": 700.0,
                "time_to_first_token_ms": 250.0,
                "decode_token_latencies_ms": [80.0, 100.0, 120.0],
            }
        ),
        encoding="utf-8",
    )
    result = parse_probe_result(path)
    assert result["prompt_eval_tps"] == pytest.approx(50.0)
    assert result["decode_tps"] == pytest.approx(10.0)
    assert result["token_latency_p50_ms"] == pytest.approx(100.0)
    assert result["token_latency_p95_ms"] == pytest.approx(118.0)


def test_summarize_repetitions_includes_sample_variance():
    summary = summarize_repetitions(
        [{"decode_tps": 10.0}, {"decode_tps": 12.0}, {"decode_tps": 14.0}]
    )
    assert summary["decode_tps"]["values"] == [10.0, 12.0, 14.0]
    assert summary["decode_tps"]["mean"] == pytest.approx(12.0)
    assert summary["decode_tps"]["variance"] == pytest.approx(4.0)


def test_cache_event_accounting_reconciles_demands(tmp_path):
    path = tmp_path / "cache.jsonl"
    events = [
        {"selected_expert_ids": list(range(8)), "hits": 2, "misses": 6,
         "bytes_transferred": 60, "blocking_duration_us": 10},
        {"selected_expert_ids": list(range(8)), "hits": 8, "misses": 0,
         "bytes_transferred": 0, "blocking_duration_us": 0},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    result = parse_cache_events(path)
    assert result == {
        "events": 2,
        "expert_demands": 16,
        "hits": 10,
        "misses": 6,
        "hit_rate": 0.625,
        "bytes_transferred": 60,
        "blocking_wall_ms": 0.01,
    }


def test_compare_modes_reports_relative_gain():
    result = compare_modes({"decode_tps": 10.0}, {"decode_tps": 11.0})
    assert result["decode_tps_percent"] == pytest.approx(10.0)
    assert result["end_to_end_percent"] is None
