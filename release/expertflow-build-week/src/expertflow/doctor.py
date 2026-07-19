"""Machine-readable hardware and toolchain preflight."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    """One measured NVIDIA GPU memory snapshot."""

    index: int
    name: str
    driver_version: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int


def parse_nvidia_smi_csv(output: str) -> tuple[GpuSnapshot, ...]:
    """Parse the stable no-header, no-units NVIDIA SMI query format."""

    snapshots: list[GpuSnapshot] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ValueError(
                f"nvidia-smi line {line_number}: expected 6 fields, got {len(fields)}"
            )
        try:
            snapshots.append(
                GpuSnapshot(
                    index=int(fields[0]),
                    name=fields[1],
                    driver_version=fields[2],
                    memory_total_mib=int(fields[3]),
                    memory_used_mib=int(fields[4]),
                    memory_free_mib=int(fields[5]),
                )
            )
        except ValueError as error:
            raise ValueError(
                f"nvidia-smi line {line_number}: invalid numeric field"
            ) from error
    return tuple(snapshots)


def tool_availability(
    names: Iterable[str],
    *,
    resolver: Callable[[str], str | None] = shutil.which,
) -> dict[str, str | None]:
    """Resolve tools without executing or mutating them."""

    return {name: resolver(name) for name in names}


def _total_ram_bytes() -> int:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.total_physical)

    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * page_count)


def collect_doctor_report(artifact_root: Path) -> dict[str, object]:
    """Collect the measured environment contract used by later runs."""

    tools = tool_availability(
        ("nvidia-smi", "cmake", "ninja", "nvcc", "gcc", "g++", "aria2c", "curl")
    )
    gpu_error: str | None = None
    gpus: tuple[GpuSnapshot, ...] = ()
    nvidia_smi = tools["nvidia-smi"]
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            gpus = parse_nvidia_smi_csv(completed.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            gpu_error = str(error)
    else:
        gpu_error = "nvidia-smi was not found"

    root = artifact_root.resolve()
    usage = shutil.disk_usage(root)
    return {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "system_ram_bytes": _total_ram_bytes(),
        "artifact_root": str(root),
        "artifact_disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
        "gpus": [asdict(gpu) for gpu in gpus],
        "gpu_error": gpu_error,
        "tools": tools,
    }
