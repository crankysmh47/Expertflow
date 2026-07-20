from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = ROOT / "docs" / "evidence" / "product-release"
DEFAULT_DEPLOYMENT = EVIDENCE_ROOT / "deployment-result.json"
RELEASE_STATE = EVIDENCE_ROOT / "release-state.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_json(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_release_evidence() -> tuple[bool, list[dict[str, object]]]:
    state = load_json(RELEASE_STATE)
    checks = []
    for item in state["evidence"]:
        path = ROOT / item["path"]
        actual = sha256_canonical_json(path) if path.is_file() else None
        checks.append({
            "path": item["path"],
            "expected_sha256": item["sha256"],
            "actual_sha256": actual,
            "match": actual == item["sha256"],
        })
    return all(item["match"] for item in checks), checks


def replay_report() -> dict[str, object]:
    state = load_json(RELEASE_STATE)
    verified, checks = verify_release_evidence()
    return {
        "schema_version": "1.0.0",
        "status": "pass" if verified else "failure",
        "mode": "measured_evidence_replay",
        "notice": "Recorded evidence replay; this command is not a live benchmark.",
        "performance": state["performance"],
        "placement": state["placement"],
        "quality": state["quality"],
        "cache_strategy": state["cache_strategy"],
        "evidence_hashes_verified": verified,
        "evidence_checks": checks,
    }


def compare_report(deployment_path: Path) -> dict[str, object]:
    deployment = load_json(deployment_path)
    quality = deployment["quality"]
    evidence = deployment.get("evidence", deployment.get("measured"))
    if not isinstance(evidence, dict):
        raise ValueError("deployment must include measured comparison evidence")
    source = evidence.get("source", deployment_path.resolve().as_posix())
    stock_tps = evidence.get("strongest_stock_decode_tps", evidence.get("stock_decode_tps"))
    if stock_tps is None:
        raise ValueError("deployment measured evidence must include stock decode TPS")
    return {
        "schema_version": "1.0.0",
        "status": "pass",
        "measurement_kind": "measured_recorded_evidence",
        "stock": {"decode_tps": stock_tps, "placement": "--cpu-moe"},
        "expertflow": {
            "decode_tps": evidence["decode_tps"],
            "peak_process_owned_mib": deployment["expected_peak_vram_mib"],
            "static_expert_layers": deployment["placement"]["static_expert_layers"],
        },
        "improvement_pct": evidence["improvement_pct"],
        "quality": quality,
        "evidence_source": source,
    }


def profile_report(model: Path) -> dict[str, object]:
    state = load_json(RELEASE_STATE)
    exists = model.is_file()
    hash_match = None
    if exists:
        hash_match = sha256_file(model) == state["model"]["sha256"]
    status = "pass" if hash_match else "warning"
    return {
        "schema_version": "1.0.0",
        "status": status,
        "measurement_kind": "measured_recorded_evidence_plus_live_path_inspection",
        "model": {"path": str(model), "exists": exists, "hash_match": hash_match},
        "routed_layer_count": 30,
        "experts_per_layer": state["placement"]["experts_per_layer"],
        "expert_bank_bytes_per_layer": state["placement"]["shadow_bytes_per_layer"],
        "cpu_expert_bottleneck_evidence": "docs/evidence/q6-placement-final/layer-profile.json",
        "recommended_static_layers": state["placement"]["layers"],
        "recommended_profile": "max-performance",
        "available_vram_mib": 16311,
        "evidence_source": "docs/evidence/q6-placement-final/results.json",
    }


def optimize_deployment(model: Path, goal: str) -> tuple[int, dict[str, object], dict[str, Any] | None]:
    if goal in {"throughput", "context", "agentic"}:
        profile = ROOT / "deployments" / f"max-{goal}.json"
        if not profile.is_file():
            return 2, {"schema_version": "1.0.0", "status": "failure", "reason": f"measured {goal} profile is not available"}, None
        deployment = load_json(profile)
    else:
        deployment = load_json(DEFAULT_DEPLOYMENT)
    deployment = json.loads(json.dumps(deployment))
    deployment["goal"] = goal
    deployment["model"]["path"] = str(model)
    return 0, {"schema_version": "1.0.0", "status": "pass", "goal": goal}, deployment


def build_runtime_command(
    deployment: Mapping[str, Any], *, runtime: Path, model: Path
) -> tuple[list[str], dict[str, str]]:
    command = [str(runtime), "-m", str(model), *[str(value) for value in deployment["arguments"]]]
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in deployment["environment"].items()})
    return command, environment


def doctor_report(model: Path | None, runtime: Path | None, server: Path | None) -> dict[str, object]:
    state = load_json(RELEASE_STATE)
    checks: list[dict[str, object]] = []
    system = platform.system()
    machine = platform.machine().lower()
    windows_x64 = system == "Windows" and machine in {"amd64", "x86_64"}
    checks.append({
        "name": "operating_system_architecture",
        "status": "pass" if windows_x64 else "warning",
        "value": {"system": system, "architecture": machine, "platform": platform.platform()},
        "next_command": "Use model-free replay: uv run expertflow demo --replay" if not windows_x64 else None,
    })
    checks.append({
        "name": "python",
        "status": "pass" if sys.version_info >= (3, 11) else "failure",
        "value": platform.python_version(),
        "next_command": "Install Python 3.11 or newer." if sys.version_info < (3, 11) else None,
    })
    uv = shutil.which("uv")
    checks.append({"name": "uv", "status": "pass" if uv else "warning", "path": uv, "next_command": None if uv else "Install uv: https://docs.astral.sh/uv/"})
    smi = shutil.which("nvidia-smi")
    gpu: dict[str, object] = {"available": False}
    if smi:
        try:
            completed = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            gpu = {"available": completed.returncode == 0, "raw": completed.stdout.strip(), "error": completed.stderr.strip()}
        except (OSError, subprocess.SubprocessError) as error:
            gpu = {"available": False, "error": str(error)}
    checks.append({"name": "nvidia_gpu_driver_vram", "status": "pass" if gpu["available"] else "warning", "value": gpu, "next_command": None if gpu["available"] else "Install a supported NVIDIA driver, or use evidence replay."})
    for name, path, expected in (
        ("model", model, state["model"]["sha256"]),
        ("llama_cli", runtime, state["runtime"]["llama_cli_sha256"]),
        ("llama_server", server, state["runtime"]["llama_server_sha256"]),
    ):
        exists = path is not None and path.is_file()
        actual = sha256_file(path) if exists else None
        size = path.stat().st_size if exists else None
        expected_size = state["model"]["bytes"] if name == "model" else None
        size_match = size == expected_size if expected_size is not None else None
        identity_match = actual == expected and size_match is not False
        architecture = "PE" if exists and path.suffix.lower() == ".exe" and path.read_bytes()[:2] == b"MZ" else ("unknown" if exists else None)
        next_command = None if identity_match else f"Set the verified {name} path and rerun expertflow doctor."
        checks.append({"name": name, "status": "pass" if identity_match else "warning", "path": str(path) if path else None, "bytes": size, "expected_bytes": expected_size, "size_match": size_match, "binary_architecture": architecture if name != "model" else None, "expected_sha256": expected, "actual_sha256": actual, "next_command": next_command})
    cuda_root = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"))
    cudart = list(cuda_root.glob("bin/cudart64_*.dll")) if system == "Windows" else []
    checks.append({"name": "cuda_runtime", "status": "pass" if cudart else "warning", "paths": [str(path) for path in cudart], "next_command": None if cudart else "Install CUDA 12.8 runtime libraries, or use evidence replay."})
    free = shutil.disk_usage(ROOT).free
    checks.append({"name": "free_disk", "status": "pass" if free >= 1024 ** 3 else "warning", "free_bytes": free, "next_command": None if free >= 1024 ** 3 else "Free at least 1 GiB for release outputs."})
    writable = os.access(EVIDENCE_ROOT, os.W_OK)
    checks.append({"name": "writable_evidence_directory", "status": "pass" if writable else "failure", "path": str(EVIDENCE_ROOT), "next_command": None if writable else "Grant write access to docs/evidence/product-release."})
    deployment_ok = all(load_json(path).get("schema_version") == "1.0.0" for path in (ROOT / "deployments").glob("max-*.json"))
    checks.append({"name": "deployment_schema", "status": "pass" if deployment_ok else "failure", "version": "1.0.0", "next_command": None if deployment_ok else "Restore the measured deployment manifests."})
    port_free = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 8080))
            port_free = True
    except OSError:
        pass
    checks.append({"name": "port_8080", "status": "pass" if port_free else "warning", "next_command": None if port_free else "Stop the process using port 8080 or pass --port <free-port>."})
    raw_gpu = str(gpu.get("raw", ""))
    verified_gpu = "RTX 5060 Ti" in raw_gpu and gpu["available"]
    verified_platform = windows_x64 and verified_gpu
    required_live = {"model", "llama_cli", "llama_server", "cuda_runtime", "deployment_schema", "writable_evidence_directory"}
    live_failures = [item for item in checks if item["name"] in required_live and item["status"] != "pass"]
    if not verified_platform:
        status, exit_code = "replay_only", 10
    elif live_failures:
        status, exit_code = "failure", 20
    else:
        status, exit_code = "pass", 0
    return {
        "schema_version": "1.0.0",
        "status": status,
        "exit_code": exit_code,
        "measurement_kind": "live_inspection",
        "replay_supported": True,
        "live_acceleration_supported": status == "pass",
        "verified_live_platform": "Windows 11 x64, NVIDIA RTX 5060 Ti 16 GB, CUDA 12.8.93",
        "release_identity": state["source"],
        "checks": checks,
    }
