from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_ENVIRONMENT_NAMES = (
    "EXPERTFLOW_LIVE_CACHE",
    "EXPERTFLOW_LIVE_CACHE_MODE",
    "EXPERTFLOW_LIVE_CACHE_LAYER",
    "EXPERTFLOW_LIVE_CACHE_LAYERS",
    "EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE",
    "EXPERTFLOW_LIVE_CACHE_NGL",
    "EXPERTFLOW_LIVE_CACHE_LOG",
    "EXPERTFLOW_LIVE_CACHE_LOG_DETAIL",
    "EXPERTFLOW_LIVE_CACHE_FORCE_EVICT",
)


def validate_manifest(manifest: dict[str, Any]) -> None:
    if set(manifest.get("prompts", {})) != {"general", "code", "translation"}:
        raise ValueError("prompts must contain exactly general, code, translation")
    if not manifest.get("modes"):
        raise ValueError("manifest requires modes")


def build_command(
    *,
    executable: Path,
    model: Path,
    output_dir: Path,
    prompt: str,
    ngl: int,
    threads: int,
    trace: bool,
) -> list[str]:
    command = [
        str(executable), "-m", str(model),
        "--tokens", str(output_dir / "tokens.json"),
        "--performance", str(output_dir / "performance.json"),
    ]
    if trace:
        command.extend(["--trace", str(output_dir / "trace.jsonl"), "--trace-mode", "full"])
    else:
        command.append("--no-trace")
    command.extend(["-n", "64", "-ngl", str(ngl), "--threads", str(threads), prompt])
    return command


def build_environment(
    inherited: dict[str, str], configured: dict[str, str]
) -> dict[str, str]:
    environment = dict(inherited)
    for name in CACHE_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    environment.update({str(key): str(value) for key, value in configured.items()})
    return environment


def _gpu_used_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one(
    *,
    command: list[str],
    cwd: Path,
    output_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[int] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            value = _gpu_used_mib()
            if value is not None:
                samples.append(value)
            stop.wait(0.1)

    before = _gpu_used_mib()
    started = time.perf_counter()
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
         (output_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.run(
            command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, text=True
        )
    stop.set()
    sampler.join(timeout=2)
    ended = time.perf_counter()
    after = _gpu_used_mib()
    artifacts = {}
    for name in ("performance.json", "tokens.json", "trace.jsonl", "cache.jsonl"):
        path = output_dir / name
        if path.exists():
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    result = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "command": command,
        "cwd": str(cwd),
        "return_code": process.returncode,
        "process_wall_seconds": ended - started,
        "gpu_system_used_before_mib": before,
        "gpu_system_peak_used_mib": max(samples) if samples else None,
        "gpu_system_used_after_mib": after,
        "gpu_sample_count": len(samples),
        "artifacts": artifacts,
    }
    (output_dir / "measurement.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output / "command-ledger.jsonl"
    for mode_name, mode in manifest["modes"].items():
        repetitions = int(mode.get("repetitions", 3))
        warmups = int(mode.get("warmups", 1))
        for domain, prompt in manifest["prompts"].items():
            for repetition in range(warmups + repetitions):
                label = "warmup" if repetition < warmups else f"rep-{repetition - warmups + 1}"
                output_dir = args.output / mode_name / domain / label
                environment = build_environment(
                    dict(os.environ), mode.get("environment", {})
                )
                if mode.get("cache"):
                    environment["EXPERTFLOW_LIVE_CACHE_LOG"] = str(output_dir / "cache.jsonl")
                command = build_command(
                    executable=Path(mode["executable"]),
                    model=Path(manifest["model"]),
                    output_dir=output_dir,
                    prompt=prompt,
                    ngl=int(mode["ngl"]),
                    threads=int(manifest.get("threads", 12)),
                    trace=bool(mode.get("trace", False)),
                )
                event = {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "mode": mode_name, "domain": domain, "repetition": label,
                    "command": command,
                }
                with ledger.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                result = run_one(
                    command=command,
                    cwd=Path(mode["cwd"]),
                    output_dir=output_dir,
                    environment=environment,
                )
                if result["return_code"] != 0:
                    return result["return_code"] or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
