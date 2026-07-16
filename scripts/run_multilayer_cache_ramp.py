from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


CACHE_ENVIRONMENT_NAMES = (
    "EXPERTFLOW_LIVE_CACHE",
    "EXPERTFLOW_LIVE_CACHE_MODE",
    "EXPERTFLOW_LIVE_CACHE_LAYER",
    "EXPERTFLOW_LIVE_CACHE_LAYERS",
    "EXPERTFLOW_LIVE_CACHE_LOG",
    "EXPERTFLOW_LIVE_CACHE_FORCE_EVICT",
)


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "threads": 12,
        "n_predict": 64,
        "ngl": 10,
        "gpu_sample_interval_ms": 100,
        "warmups": 1,
        "repetitions": 3,
    }
    for key, expected in required.items():
        if int(manifest.get(key, -1)) != expected:
            raise ValueError(f"{key} must be exactly {expected}")
    if set(manifest.get("prompts", {})) != {"general", "code", "translation"}:
        raise ValueError("prompts must contain exactly general, code, translation")
    layers = manifest.get("layers")
    if not layers or layers != sorted(set(layers)):
        raise ValueError("layers must be a non-empty ascending unique list")
    if set(manifest.get("modes", {})) != {"cache_off", "cache_on"}:
        raise ValueError("modes must contain exactly cache_off and cache_on")
    if not Path(manifest["model"]).is_absolute():
        raise ValueError("model path must be absolute")
    for mode in manifest["modes"].values():
        if not Path(mode["executable"]).is_absolute():
            raise ValueError("executable paths must be absolute")


def build_cache_environment(
    inherited: dict[str, str],
    *,
    layers: tuple[int, ...],
    event_path: Path,
) -> dict[str, str]:
    if not event_path.is_absolute():
        raise ValueError("cache event path must be absolute")
    environment = dict(inherited)
    for name in CACHE_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    environment.update(
        {
            "EXPERTFLOW_LIVE_CACHE": "1",
            "EXPERTFLOW_LIVE_CACHE_MODE": "blocking",
            "EXPERTFLOW_LIVE_CACHE_LAYERS": ",".join(str(layer) for layer in layers),
            "EXPERTFLOW_LIVE_CACHE_LOG": str(event_path),
        }
    )
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    layers = tuple(int(layer) for layer in manifest["layers"])
    transformed = {
        "schema_version": "1.0.0",
        "model": manifest["model"],
        "threads": manifest["threads"],
        "prompts": manifest["prompts"],
        "modes": {},
    }
    for name, mode in manifest["modes"].items():
        transformed_mode = {
            "executable": mode["executable"],
            "cwd": mode.get("cwd", str(Path(mode["executable"]).parent)),
            "ngl": manifest["ngl"],
            "trace": True,
            "warmups": manifest["warmups"],
            "repetitions": manifest["repetitions"],
        }
        if name == "cache_on":
            transformed_mode["cache"] = True
            transformed_mode["environment"] = {
                "EXPERTFLOW_LIVE_CACHE": "1",
                "EXPERTFLOW_LIVE_CACHE_MODE": "blocking",
                "EXPERTFLOW_LIVE_CACHE_LAYERS": ",".join(
                    str(layer) for layer in layers
                ),
            }
        transformed["modes"][name] = transformed_mode
    transformed_path = args.output / "resolved-benchmark-manifest.json"
    transformed_path.write_text(
        json.dumps(transformed, indent=2, sort_keys=True), encoding="utf-8"
    )
    command = [
        os.fspath(Path(__file__).with_name("run_performance_diagnostic.py")),
        "--manifest",
        os.fspath(transformed_path),
        "--output",
        os.fspath(args.output),
    ]
    return subprocess.call([os.fspath(Path(os.sys.executable)), *command])


if __name__ == "__main__":
    raise SystemExit(main())
