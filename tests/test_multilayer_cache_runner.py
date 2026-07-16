import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "multilayer_runner", Path("scripts/run_multilayer_cache_ramp.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_manifest_requires_frozen_protocol_and_absolute_paths(tmp_path):
    manifest = {
        "model": str(tmp_path / "model.gguf"),
        "threads": 12,
        "n_predict": 64,
        "ngl": 10,
        "gpu_sample_interval_ms": 100,
        "warmups": 1,
        "repetitions": 3,
        "layers": [0, 24],
        "prompts": {"general": "g", "code": "c", "translation": "t"},
        "modes": {
            "cache_off": {"executable": str(tmp_path / "probe.exe")},
            "cache_on": {"executable": str(tmp_path / "probe.exe")},
        },
    }
    runner.validate_manifest(manifest)
    for key, value in {
        "ngl": 9,
        "n_predict": 63,
        "warmups": 0,
        "repetitions": 2,
        "gpu_sample_interval_ms": 200,
    }.items():
        invalid = dict(manifest)
        invalid[key] = value
        with pytest.raises(ValueError):
            runner.validate_manifest(invalid)


def test_cache_environment_uses_plural_layers_and_absolute_event_path(tmp_path):
    environment = runner.build_cache_environment(
        {"PATH": "x", "EXPERTFLOW_LIVE_CACHE_LAYER": "24"},
        layers=(0, 24),
        event_path=(tmp_path / "cache.jsonl").resolve(),
    )
    assert environment["EXPERTFLOW_LIVE_CACHE"] == "1"
    assert environment["EXPERTFLOW_LIVE_CACHE_MODE"] == "blocking"
    assert environment["EXPERTFLOW_LIVE_CACHE_LAYERS"] == "0,24"
    assert Path(environment["EXPERTFLOW_LIVE_CACHE_LOG"]).is_absolute()
    assert "EXPERTFLOW_LIVE_CACHE_LAYER" not in environment
