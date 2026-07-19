from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
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


def verify_release_evidence() -> tuple[bool, list[dict[str, object]]]:
    state = load_json(RELEASE_STATE)
    checks = []
    for item in state["evidence"]:
        path = ROOT / item["path"]
        actual = sha256_file(path) if path.is_file() else None
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
    checks.append({"name": "windows", "status": "pass" if os.name == "nt" else "failure", "value": platform.platform()})
    smi = shutil.which("nvidia-smi")
    gpu: dict[str, object] = {"available": False}
    if smi:
        completed = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        gpu = {"available": completed.returncode == 0, "raw": completed.stdout.strip(), "error": completed.stderr.strip()}
    checks.append({"name": "nvidia_gpu", "status": "pass" if gpu["available"] else "failure", "value": gpu})
    for name, path, expected in (
        ("model", model, state["model"]["sha256"]),
        ("llama_cli", runtime, state["runtime"]["llama_cli_sha256"]),
        ("llama_server", server, state["runtime"]["llama_server_sha256"]),
    ):
        exists = path is not None and path.is_file()
        actual = sha256_file(path) if exists else None
        checks.append({"name": name, "status": "pass" if actual == expected else "warning", "path": str(path) if path else None, "expected_sha256": expected, "actual_sha256": actual})
    cudart = list(Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")).glob("bin/cudart64_*.dll"))
    checks.append({"name": "cuda_runtime_dll", "status": "pass" if cudart else "warning", "paths": [str(path) for path in cudart]})
    severity = "failure" if any(item["status"] == "failure" for item in checks) else ("warning" if any(item["status"] == "warning" for item in checks) else "pass")
    return {"schema_version": "1.0.0", "status": severity, "measurement_kind": "live_inspection", "checks": checks}
