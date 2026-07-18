from __future__ import annotations

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "q6_selected_runner", Path("scripts/run_q6_selected_static.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_pair_order_alternates_which_mode_runs_first() -> None:
    assert runner.pair_order(4) == [
        ("off", "on"),
        ("on", "off"),
        ("off", "on"),
        ("on", "off"),
    ]


def test_environment_is_default_off_and_four_layer_on() -> None:
    inherited = {
        "PATH": "cuda",
        "LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER": "99",
        "LLAMA_EXPERTFLOW_SPLIT_PROFILE": "profile.json",
    }

    off = runner.build_environment(inherited, "off")
    on = runner.build_environment(inherited, "on")

    assert "LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER" not in off
    assert "LLAMA_EXPERTFLOW_SPLIT_PROFILE" not in off
    assert off["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert on["LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER"] == "0,1,15,20"
    assert "LLAMA_EXPERTFLOW_SPLIT_PROFILE" not in on
    assert on["GGML_CUDA_DISABLE_GRAPHS"] == "1"


def test_server_command_freezes_matched_q6_configuration(tmp_path: Path) -> None:
    command = runner.build_server_command(
        Path("llama-server.exe"), Path("model.gguf"), 18080, tmp_path / "runtime.log"
    )

    assert command[:3] == ["llama-server.exe", "-m", "model.gguf"]
    for expected in (["-ngl", "99"], ["-c", "2048"], ["-b", "2048"], ["-ub", "512"], ["-t", "12"]):
        index = command.index(expected[0])
        assert command[index : index + 2] == expected
    assert "--cpu-moe" in command
    assert "--no-warmup" not in command


def test_runtime_environment_includes_binary_and_cuda_dll_directories(tmp_path: Path) -> None:
    executable = tmp_path / "bin" / "llama-server.exe"
    inherited = {"PATH": "existing", "CUDA_PATH": str(tmp_path / "cuda")}

    environment = runner.runtime_environment(inherited, "off", executable)

    assert environment["PATH"].split(";")[:2] == [
        str(executable.parent.resolve()),
        str((tmp_path / "cuda" / "bin").resolve()),
    ]


def test_stream_metrics_use_first_and_inter_token_arrivals() -> None:
    events = [
        (10.100, {"content": "a", "tokens": [1]}),
        (10.140, {"content": "b", "tokens": [2]}),
        (10.200, {"content": "", "stop": True, "timings": {"predicted_per_second": 25.0}}),
    ]

    metrics = runner.stream_metrics(10.000, events)

    assert metrics["ttft_ms"] == 100.0
    assert metrics["token_ids"] == [1, 2]
    assert metrics["token_latency_ms"] == [40.0]
    assert metrics["decode_tps"] == 25.0
