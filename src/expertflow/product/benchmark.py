"""Bounded local measurements with raw artifacts and process-owned memory."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from .server import ServerSession

TRAIN_PROMPT = "Explain how to run a local language model efficiently. Give a detailed numbered list of practical steps."
HELDOUT_PROMPTS = ["Describe the tradeoffs between CPU RAM and GPU VRAM for local inference in detail.",
                   "Write a Python function to merge two sorted lists, then explain its time complexity and edge cases."]


def capture_performance_host(profile: dict, *, deadline=None) -> dict:
    from expertflow.doctor import collect_doctor_report
    hardware = collect_doctor_report(Path(profile["model"]["path"]).parent)
    gpus = [{k: gpu[k] for k in ["index", "name", "driver_version", "memory_total_mib"]} for gpu in hardware["gpus"]]
    if profile.get("host", {}).get("gpus") != gpus:
        raise ValueError("GPU/driver identity changed since setup; create a new profile before measuring.")
    if any(gpu["memory_free_mib"] < 256 for gpu in hardware["gpus"]):
        raise ValueError("GPU free-memory reserve unavailable; idle other GPU applications and retry in a fresh job.")
    if hardware["gpus"]:
        idle_deadline = min(time.monotonic()+5, deadline if deadline is not None else float('inf'))
        consecutive = 0
        for _ in range(20):
            left=idle_deadline-time.monotonic()
            if left <= 0:
                break
            try:
                utilization = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                             capture_output=True, text=True, timeout=left, check=True)
            except (subprocess.SubprocessError, OSError) as error:
                raise ValueError("GPU idle probe failed; check driver/nvidia-smi before measuring.") from error
            try:
                values = [int(line.strip()) for line in utilization.stdout.strip().splitlines()]
            except ValueError as error:
                raise ValueError("GPU idle state unavailable; measurement is unverified.") from error
            if len(values) != len(gpus) or any(not 0 <= value <= 100 for value in values):
                raise ValueError("GPU idle state unavailable; measurement is unverified.")
            consecutive = consecutive+1 if all(value <= 5 for value in values) else 0
            if consecutive == 3:
                break
            time.sleep(min(0.2,max(0,idle_deadline-time.monotonic())))
        if consecutive < 3:
            raise ValueError("GPU is busy; idle other GPU applications before a fresh measurement. No application was closed.")
    result = {"gpus": gpus, "system_ram_bytes": hardware["system_ram_bytes"]}
    if os.name == "nt":
        from expertflow.compiler.preflight import capture_host_environment
        result["controls"] = capture_host_environment()
    return result


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_completion(result: dict, prompt: str, *, expected_predict=None) -> dict:
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Runtime did not return nonempty token IDs; measurement is unverified.")
    timings = result.get("timings", {})
    if not isinstance(timings, dict):
        raise ValueError("Invalid runtime timing object; measurement is unverified.")
    tps = timings.get("predicted_per_second")
    if type(tps) not in {int, float} or not math.isfinite(tps) or tps <= 0:
        raise ValueError("Invalid decode timing/TPS; measurement is unverified.")
    for key in ["prompt_n", "predicted_n"]:
        if type(timings.get(key)) is not int or timings[key] < 1:
            raise ValueError(f"Missing/invalid workload count: {key}.")
    for key in ["prompt_ms", "predicted_ms"]:
        value = timings.get(key)
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Missing/invalid timing: {key}.")
    if len(tokens) != timings["predicted_n"]:
        raise ValueError("Generated token IDs differ from the runtime token count.")
    if expected_predict is not None and timings["predicted_n"] != expected_predict:
        raise ValueError("Incomplete fixed-length generation; registered token count not met.")
    return {"tokens": tokens, "decode_tps": tps, "prompt_tokens": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"), "decode_ms": timings.get("predicted_ms"),
            "predicted_tokens": timings.get("predicted_n"), "wall_seconds": result.get("wall_seconds"),
            "ttft_seconds": result.get("ttft_seconds"), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


class MemoryMonitor:
    def __init__(self, pid):
        self.pid, self.samples, self.errors = pid, [], []
        self.stop = threading.Event()
        self.sampler = None
        self.thread = None

    def __enter__(self):
        from expertflow.compiler.runner import WindowsGpuMemorySampler
        if os.name == "nt":
            query = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=10, check=True)
            uuids = query.stdout.strip().splitlines()
            if len(uuids) != 1:
                raise ValueError("Measured local tuning currently qualifies one NVIDIA GPU only.")
            self.sampler = WindowsGpuMemorySampler(uuids[0].strip())
        def sample():
            while not self.stop.is_set():
                try:
                    if self.sampler:
                        self.samples.append({"time": time.monotonic(), **self.sampler(self.pid)})
                    else:
                        # CPU/platform runs are useful diagnostics, not GPU tuning qualification.
                        from expertflow.runtime.measurement import _process_memory
                        self.samples.append({"time": time.monotonic(), "pid": self.pid, "cpu": _process_memory(self.pid)})
                except Exception as error:
                    self.errors.append(str(error))
                self.stop.wait(0.1)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=15)
        if self.sampler:
            self.sampler.close()


def measure_profile(profile: dict, output: Path, *, prompts=None, predict=256,
                    repeats=3, deadline=None) -> dict:
    prompts = [TRAIN_PROMPT] if prompts is None else prompts
    if type(repeats) is not int or type(predict) is not int or not 1 <= repeats <= 20 or not 1 <= predict <= 65536 or not prompts or len(prompts) > 10 or any(not isinstance(p, str) or not p or len(p.encode()) > 1024**2 for p in prompts):
        raise ValueError("Invalid benchmark repetition/token/prompt limits.")
    if output.exists():
        raise ValueError("Measurement directory exists; use a fresh output.")
    output.mkdir(parents=True)
    deadline = time.monotonic() + 600 if deadline is None else deadline
    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Declared wall budget exhausted; baseline retained.")
        return left
    records = []
    result = {"status": "INCONCLUSIVE", "records": records, "memory": {}, "cleanup": False}
    write_json(output / "profile-input.json", profile)
    server = None
    try:
        result["host_environment"] = capture_performance_host(profile, deadline=deadline)
        with ServerSession(profile, log_dir=output, health_timeout=min(180, remaining()), deadline=deadline) as server:
            with MemoryMonitor(server.child.process.pid) as monitor:
                server.complete(TRAIN_PROMPT, predict=16, timeout=min(120, remaining()), ignore_eos=True)
                for repetition in range(repeats):
                    for index, prompt in enumerate(prompts):
                        response = server.complete(prompt, predict=predict, timeout=min(120, remaining()), ignore_eos=True, stream_measure=True)
                        write_json(output / f"response-{repetition}-{index}.json", response)
                        records.append(parse_completion(response, prompt, expected_predict=predict))
                time.sleep(min(0.2, remaining()))
            result["memory"] = {"samples": monitor.samples, "errors": monitor.errors,
                "peak_owned_bytes": max((s.get("dedicated_bytes", 0) for s in monitor.samples), default=0),
                "scope": "process-owned PDH/NVML samples after healthy load; not a load-time allocation peak"}
            result["props"] = server.props
        result["cleanup"] = server.child.process.poll() is not None
        if capture_performance_host(profile, deadline=deadline) != result["host_environment"]:
            raise ValueError("Host controls changed during measurement; retain baseline.")
        if monitor.errors:
            raise ValueError("Memory telemetry failed; performance is unverified.")
        if os.name == "nt" and (not any(s.get("dedicated_bytes", 0) > 0 for s in monitor.samples) or
                                any(s.get("device_free_bytes", 0) < 256 * 1024**2 for s in monitor.samples if s.get("counter_available"))):
            raise ValueError("Owned GPU allocation/reserve could not be verified.")
        result.update(status="MEASURED", decode_tps=statistics.mean(r["decode_tps"] for r in records),
                      tokens=[t for r in records for t in r["tokens"]],
                      prompt_tokens=[r["prompt_tokens"] for r in records],
                      ttft_seconds=statistics.mean(r["ttft_seconds"] for r in records if r["ttft_seconds"] is not None) if any(r["ttft_seconds"] is not None for r in records) else None)
        return result
    except BaseException as error:
        result["reason"] = str(error) or type(error).__name__
        if server and server.child:
            result["cleanup"] = server.child.process.poll() is not None
        raise
    finally:
        result["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name != "measurement.json"}
        write_json(output / "measurement.json", result)


def bench_profile(profile: dict, output: Path, *, predict=256, repeats=3, budget_seconds=600, prompts=None) -> dict:
    if output.exists():
        raise ValueError("Job output exists; choose a fresh directory.")
    if type(budget_seconds) not in {int, float} or not math.isfinite(budget_seconds) or not 1 <= budget_seconds <= 86400:
        raise ValueError("Benchmark wall budget must be 1–86400 seconds.")
    output.mkdir(parents=True)
    from .reports import freeze_sources
    registration = {"kind": "local-benchmark", "maximum_model_processes": 1, "profile": profile,
        "predict": predict, "repeats": repeats, "prompts": prompts or [TRAIN_PROMPT],
        "budget_seconds": budget_seconds, "warmup_predict": 16,
        "maximum_external_gpu_utilization_pct": 5,
        "idle_settle_seconds":5, "consecutive_idle_samples":3,
        "sources": freeze_sources(output)}
    write_json(output / "registration.json", registration)
    start = time.monotonic()
    result = {"status": "INCONCLUSIVE", "model_processes": 0, "model_attempts": 1, "registration_sha256": hashlib.sha256((output / "registration.json").read_bytes()).hexdigest()}
    try:
        result["measurement"] = measure_profile(profile, output / "measurement", predict=predict,
                                                repeats=repeats, prompts=prompts, deadline=start + budget_seconds)
        result.update(status="MEASURED", reason="Diagnostic benchmark on this profile/workload; no improvement claim.")
    except (ValueError, OSError, RuntimeError, TimeoutError, KeyboardInterrupt) as error:
        result["reason"] = str(error) or "Cancelled; partial evidence retained."
    finally:
        result["model_processes"] = len(list(output.glob("*/launch.json")))
        result["wall_seconds"] = time.monotonic() - start
        write_json(output / "report.json", result)
    return result
