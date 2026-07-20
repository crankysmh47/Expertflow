import pytest

from expertflow.analysis.q6_routing import (
    build_workloads,
    merged_workload_ids,
    probe_command,
    simulate_hybrid_candidate,
    summarize_routing_locality,
)
from expertflow.trace.schema import RouterTraceEvent


def event(sequence: str, token: int, layer: int, experts: tuple[int, ...]) -> RouterTraceEvent:
    return RouterTraceEvent(
        schema_version="1.0.0",
        request_id=f"request-{sequence}",
        conversation_id=sequence,
        turn_index=0,
        phase="decode",
        forward_id=token,
        hook_order=token * 30 + layer,
        token_index=token,
        token_id=100 + token,
        layer_id=layer,
        selected_expert_ids=experts,
        selected_expert_weights=None,
        observed_at_ns=token * 1_000 + layer,
    )


def test_builds_all_frozen_workload_families_deterministically() -> None:
    mmlu = [
        {"question": "Q1?", "choices": ["a", "b", "c", "d"], "answer": 0,
         "subject": "s", "row_id": 1, "selection_sha256": "a" * 64},
        {"question": "Q2?", "choices": ["a", "b", "c", "d"], "answer": 1,
         "subject": "s", "row_id": 2, "selection_sha256": "b" * 64},
    ]
    conversations = [
        {"conversation_id": "train-general", "split": "train", "domain": "general", "prompt": "g"},
        {"conversation_id": "validation-general", "split": "validation", "domain": "general", "prompt": "v"},
        {"conversation_id": "test-general", "split": "test", "domain": "general", "prompt": "t"},
    ]

    workloads = build_workloads(
        ppl_text="0123456789" * 100,
        mmlu_items=mmlu,
        conversations=conversations,
        ppl_window_count=2,
        ppl_window_chars=20,
    )

    assert [row.workload_family for row in workloads].count("performance_512") == 1
    assert [row.workload_family for row in workloads].count("heldout_ppl") == 2
    assert [row.workload_family for row in workloads].count("fixed_mmlu") == 2
    assert [row.workload_family for row in workloads].count("representative_conversation") == 3
    assert next(row for row in workloads if row.workload_family == "performance_512").n_predict == 512
    assert [row.source_offset for row in workloads if row.workload_family == "heldout_ppl"] == [0, 980]
    assert all(row.n_predict == 1 for row in workloads if row.workload_family in {"heldout_ppl", "fixed_mmlu"})


def test_reports_per_layer_locality_and_lru_bytes() -> None:
    traces = {
        "seq-a": [
            event("seq-a", 0, 4, (1, 2)),
            event("seq-a", 1, 4, (1, 3)),
            event("seq-a", 2, 4, (1, 2)),
        ],
        "seq-b": [event("seq-b", 0, 4, (2, 3))],
    }

    report = summarize_routing_locality(
        traces,
        capacities=(2, 3),
        expert_bundle_bytes=100,
    )

    layer = report[4]
    assert layer["event_count"] == 4
    assert layer["expert_demand_count"] == 8
    assert layer["working_set_size"] == 3
    assert layer["unique_experts_per_sequence"] == {"min": 2, "max": 3, "mean": 2.5}
    assert layer["adjacent_token_reuse_rate"] == 0.5
    assert layer["temporal_reuse_rate"] == 0.375
    assert layer["lru"]["2"]["hits"] == 2
    assert layer["lru"]["2"]["misses"] == 6
    assert layer["lru"]["2"]["misses_per_token"] == 1.5
    assert layer["lru"]["2"]["h2d_bytes_per_token"] == 150.0
    assert layer["lru"]["3"]["hits"] == 3
    assert layer["frequency"]["1"] == 3
    assert layer["reuse_distance"]["cold"] == 5


def test_probe_command_is_cache_disabled_and_bounded() -> None:
    workload = build_workloads(
        ppl_text="x" * 100,
        mmlu_items=[],
        conversations=[],
        ppl_window_count=1,
        ppl_window_chars=20,
    )[0]

    command = probe_command(
        workload,
        probe="probe.exe",
        model="model.gguf",
        tokens="tokens.json",
        trace="trace.raw.jsonl",
    )

    assert command[:3] == ["probe.exe", "-m", "model.gguf"]
    assert command[command.index("-n") + 1] == "512"
    assert command[command.index("-ngl") + 1] == "10"
    assert "--trace-mode" in command and "full" in command
    assert not {"-b", "-ub", "--seed", "--temp", "--ignore-eos"} & set(command)
    assert not any("cache" in argument.lower() or "predict" in argument.lower() for argument in command)


def test_resume_merges_workload_families_without_losing_prior_ids() -> None:
    workloads = build_workloads(
        ppl_text="x" * 100,
        mmlu_items=[],
        conversations=[],
        ppl_window_count=1,
        ppl_window_chars=20,
    )

    assert merged_workload_ids([{"workload_id": "older"}], workloads) == (
        "older",
        "performance-512",
        "ppl-window-01",
    )


def test_simulates_hybrid_memory_and_empirical_runtime_cost() -> None:
    result = simulate_hybrid_candidate(
        static_tps=25.0,
        cached_layers=(1, 2),
        capacity=96,
        full_slots=128,
        shadow_bytes_per_layer=100,
        misses_per_token={1: 0.1, 2: 0.2},
        transfer_ms_per_miss=1.0,
        intrinsic_overhead_fraction_per_layer=0.01,
    )

    assert result["freed_bytes"] == 50
    assert result["blocking_ms_per_token"] == pytest.approx(0.3)
    assert result["intrinsic_overhead_ms_per_token"] == pytest.approx(0.8)
    assert result["projected_tps"] == pytest.approx(1000 / 41.1)
    assert result["retained_tps_fraction"] == pytest.approx((1000 / 41.1) / 25.0)
