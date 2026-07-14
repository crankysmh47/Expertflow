"""Measured execution wrapper for the unmodified llama.cpp baseline."""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from expertflow.runtime.baseline import BaselineRunConfig, build_llama_command


def parse_compute_apps_csv(output: str) -> dict[int, int]:
    """Return total NVIDIA compute memory in MiB for each process ID."""

    usage: dict[int, int] = {}
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise ValueError(
                f"nvidia-smi compute line {line_number}: expected 2 fields"
            )
        try:
            process_id = int(fields[0])
            if fields[1] == "[N/A]":
                continue
            memory_mib = int(fields[1])
        except ValueError as error:
            raise ValueError(
                f"nvidia-smi compute line {line_number}: invalid integer"
            ) from error
        usage[process_id] = usage.get(process_id, 0) + memory_mib
    return usage


def _run_nvidia_smi(arguments: list[str]) -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise FileNotFoundError("nvidia-smi was not found")
    completed = subprocess.run(
        [executable, *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout


def _gpu_memory_snapshot() -> dict[int, dict[str, int]]:
    output = _run_nvidia_smi(
        [
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    result: dict[int, dict[str, int]] = {}
    for line_number, line in enumerate(output.splitlines(), start=1):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(
                f"nvidia-smi GPU line {line_number}: expected 4 fields"
            )
        index, total, used, free = (int(field) for field in fields)
        result[index] = {
            "total_mib": total,
            "used_mib": used,
            "free_mib": free,
        }
    return result


def _process_gpu_memory_mib(process_id: int) -> int:
    try:
        output = _run_nvidia_smi(
            [
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        return 0
    return parse_compute_apps_csv(output).get(process_id, 0)


def _windows_process_memory(process_id: int) -> dict[str, int] | None:
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    process_query_information = 0x0400
    process_vm_read = 0x0010
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(
        process_query_information | process_vm_read, False, process_id
    )
    if not handle:
        return None
    try:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not success:
            return None
        return {
            "working_set_bytes": int(counters.working_set_size),
            "peak_working_set_bytes": int(counters.peak_working_set_size),
            "private_bytes": int(counters.private_usage),
            "peak_pagefile_bytes": int(counters.peak_pagefile_usage),
        }
    finally:
        kernel32.CloseHandle(handle)


def _process_memory(process_id: int) -> dict[str, int] | None:
    if os.name == "nt":
        return _windows_process_memory(process_id)
    status = Path(f"/proc/{process_id}/status")
    if not status.exists():
        return None
    values: dict[str, int] = {}
    for line in status.read_text(encoding="utf-8").splitlines():
        key, _, raw = line.partition(":")
        if key in {"VmRSS", "VmHWM", "VmSize", "VmPeak"}:
            values[key] = int(raw.strip().split()[0]) * 1024
    return {
        "working_set_bytes": values.get("VmRSS", 0),
        "peak_working_set_bytes": values.get("VmHWM", 0),
        "private_bytes": values.get("VmSize", 0),
        "peak_pagefile_bytes": values.get("VmPeak", 0),
    }


def run_measured_baseline(
    config: BaselineRunConfig,
    *,
    model_sha256: str,
    manifest_path: Path,
    sample_interval_seconds: float = 0.5,
) -> dict[str, object]:
    """Run llama.cpp and persist raw output plus full memory evidence."""

    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if not config.executable.is_file():
        raise FileNotFoundError(config.executable)
    if not config.model.is_file():
        raise FileNotFoundError(config.model)

    run_dir = manifest_path.resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    command = build_llama_command(config)

    before_gpu = _gpu_memory_snapshot()
    started_at = datetime.now(timezone.utc)
    started_ns = time.perf_counter_ns()
    peak_process_gpu_mib = 0
    peak_gpu_used_mib = {
        index: snapshot["used_mib"] for index, snapshot in before_gpu.items()
    }
    peak_process_memory = {
        "working_set_bytes": 0,
        "peak_working_set_bytes": 0,
        "private_bytes": 0,
        "peak_pagefile_bytes": 0,
    }

    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
        )
        while process.poll() is None:
            memory = _process_memory(process.pid)
            if memory:
                for key, value in memory.items():
                    peak_process_memory[key] = max(
                        peak_process_memory[key], value
                    )
            try:
                gpu = _gpu_memory_snapshot()
                for index, snapshot in gpu.items():
                    peak_gpu_used_mib[index] = max(
                        peak_gpu_used_mib.get(index, 0), snapshot["used_mib"]
                    )
                peak_process_gpu_mib = max(
                    peak_process_gpu_mib,
                    _process_gpu_memory_mib(process.pid),
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(sample_interval_seconds)
        return_code = process.wait()

    elapsed_ns = time.perf_counter_ns() - started_ns
    finished_at = datetime.now(timezone.utc)
    after_gpu = _gpu_memory_snapshot()
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "status": "passed" if return_code == 0 else "failed",
        "return_code": return_code,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_ns": elapsed_ns,
        "command": command,
        "model": {
            "path": str(config.model.resolve()),
            "size_bytes": config.model.stat().st_size,
            "sha256": model_sha256,
        },
        "runtime": {
            "path": str(config.executable.resolve()),
        },
        "outputs": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "runtime_log": str(config.log_file.resolve()),
        },
        "memory": {
            "gpu_before": before_gpu,
            "gpu_after": after_gpu,
            "peak_gpu_used_mib": peak_gpu_used_mib,
            "peak_process_gpu_mib": peak_process_gpu_mib,
            "peak_process": peak_process_memory,
        },
    }
    manifest_path.resolve().write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
