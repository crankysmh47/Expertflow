from pathlib import Path


def test_router_probe_exposes_machine_readable_performance_output():
    source = Path("native/router_probe/main.cpp").read_text(encoding="utf-8")
    assert "--performance" in source
    assert "llama_perf_context(context)" in source
    assert '\\"prompt_eval_ms\\"' in source
    assert '\\"decode_token_latencies_ms\\"' in source
    assert "time_to_first_token_ms" in source
