from __future__ import annotations

from scripts.summarize_q1b_generation import parse_perf_line, routing_overlap


def test_parse_cli_performance_line():
    assert parse_perf_line("[ Prompt: 24.1 t/s | Generation: 26.8 t/s ]") == (24.1, 26.8)


def test_routing_overlap_distinguishes_set_and_order():
    reference = {("decode", 1, 2): [1, 2, 3]}
    reordered = {("decode", 1, 2): [3, 2, 1]}
    result = routing_overlap(reference, reordered)
    assert result["set_overlap_rate"] == 1.0
    assert result["order_overlap_rate"] == 0.0
