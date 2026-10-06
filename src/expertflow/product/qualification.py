"""Portable, deliberately narrow scheduling policy backed by archived source audits."""
from __future__ import annotations
import json
from pathlib import Path
import platform


def eligible_threads(profile: dict) -> list[int]:
    policy = json.loads(Path(__file__).with_name("stock-policy.json").read_text())
    if platform.system() != "Windows" or platform.machine() not in {"AMD64", "x86_64"}:
        return []
    model = profile["model"]
    scope = next((scope for scope in policy["models"] if scope["sha256"] == model.get("sha256") and scope["bytes"] == model.get("bytes")), None)
    if scope is None:
        return []
    runtime = profile["runtime"]
    files = {Path(p).name: digest for p, digest in runtime.get("files", {}).items()}
    manifest = policy["runtime"]
    if any(files.get(name) != digest for name, digest in {**manifest["binaries"], **manifest["dependencies"]}.items()):
        return []
    if not any(files.get(name) == manifest["cuda_runtime_sha256"] for name in files if name.startswith("cudart")):
        return []
    settings = profile["settings"]
    if settings.get("gpu_layers") != 99 or settings.get("cpu_moe") is not scope["cpu_moe"] or any(settings.get(k) != "f16" for k in ["kv_type_k", "kv_type_v"]):
        return []
    return [8, 12, 16]
