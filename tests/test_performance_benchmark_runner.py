import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "performance_runner", Path("scripts/run_performance_diagnostic.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_build_command_uses_fixed_deterministic_contract(tmp_path):
    command = runner.build_command(
        executable=Path("probe.exe"),
        model=Path("model.gguf"),
        output_dir=tmp_path,
        prompt="hello",
        ngl=10,
        threads=12,
        trace=True,
    )
    assert command[:3] == ["probe.exe", "-m", "model.gguf"]
    assert ["-n", "64"] == command[command.index("-n"):command.index("-n") + 2]
    assert ["-ngl", "10"] == command[command.index("-ngl"):command.index("-ngl") + 2]
    assert "--trace-mode" in command
    assert command[-1] == "hello"


def test_validate_manifest_requires_three_domains():
    with pytest.raises(ValueError, match="general, code, translation"):
        runner.validate_manifest({"prompts": {"general": "x"}})


def test_build_environment_removes_inherited_cache_settings():
    result = runner.build_environment(
        {
            "PATH": "x",
            "EXPERTFLOW_LIVE_CACHE": "1",
            "EXPERTFLOW_LIVE_CACHE_LAYERS": "0,24",
            "EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE": "1",
            "EXPERTFLOW_LIVE_CACHE_NGL": "10",
            "EXPERTFLOW_LIVE_CACHE_LOG_DETAIL": "aggregate",
        },
        {"CUSTOM": "value"},
    )
    assert "EXPERTFLOW_LIVE_CACHE" not in result
    assert "EXPERTFLOW_LIVE_CACHE_LAYERS" not in result
    assert "EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE" not in result
    assert "EXPERTFLOW_LIVE_CACHE_NGL" not in result
    assert "EXPERTFLOW_LIVE_CACHE_LOG_DETAIL" not in result
    assert result["CUSTOM"] == "value"
