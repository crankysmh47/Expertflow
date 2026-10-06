"""Versioned portable stock launch profiles, distinct from compiler plans."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import tempfile
import uuid
from .models import estimate_memory
from .runtime import file_digest, verify_runtime


def create_profile(model: dict, runtime: dict, *, context: int, gpu_layers=None,
                   threads=None, cpu_moe=False, name=None) -> dict:
    if model.get("context_limit") and context > model["context_limit"]:
        raise ValueError(f"Requested context {context} exceeds model context {model['context_limit']}; choose an explicit supported context.")
    for label, value, minimum in [("GPU layers", gpu_layers, 0), ("threads", threads, 1)]:
        if value is not None and (type(value) is not int or value < minimum or value > 65536):
            raise ValueError(f"Invalid {label} value.")
    return {"schema_version": 1, "kind": "stock-launch-profile", "id": uuid.uuid4().hex,
            "name": name or model["name"], "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "RUNNABLE-UNTUNED", "load_verified": False,
            "model": model, "runtime": runtime,
            "host": {"system": platform.system(), "machine": platform.machine(), "cpu_count": os.cpu_count()},
            "settings": {"context": context, "gpu_layers": gpu_layers, "threads": threads,
                         "cpu_moe": bool(cpu_moe), "seed": 42, "temperature": 0.0,
                         "kv_type_k": "f16", "kv_type_v": "f16"},
            "memory_estimate": estimate_memory(model, context), "evidence": []}


def save_profile(path: Path, profile: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    descriptor, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            revision = path.with_name(path.name + "." + uuid.uuid4().hex + ".bak")
            revision.write_bytes(path.read_bytes())
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def load_profile(path: Path) -> dict:
    try:
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError("Oversized profile.")
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("kind") != "stock-launch-profile":
            raise ValueError("Unsupported profile schema/kind; regenerate with local setup.")
        for key in ["model", "runtime", "settings", "host"]:
            if not isinstance(result.get(key), dict):
                raise ValueError(f"Invalid profile {key}.")
        settings = result["settings"]
        estimate_memory(result["model"], settings["context"])
        for key in ["gpu_layers", "threads"]:
            v = settings[key]
            if v is not None and (type(v) is not int or not (0 if key == "gpu_layers" else 1) <= v <= 65536):
                raise ValueError(f"Invalid profile {key}.")
        if type(settings["cpu_moe"]) is not bool or settings["seed"] != 42 or settings["temperature"] != 0.0 or any(settings[k] != "f16" for k in ["kv_type_k", "kv_type_v"]):
            raise ValueError("Unsupported profile numerical settings; regenerate with local setup.")
        if not isinstance(result["model"]["files"], list) or not result["model"]["files"]:
            raise ValueError("Invalid profile model files.")
        return result
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read profile: {error}") from error


def verify_profile(profile: dict, *, deadline=None) -> None:
    try:
        _verify_profile(profile, deadline=deadline)
    except TimeoutError:
        raise
    except (KeyError, TypeError, AttributeError, OSError) as error:
        raise ValueError(f"Invalid profile/artifact identity; run setup again: {error}") from error


def _verify_profile(profile: dict, *, deadline=None) -> None:
    if profile.get("status") == "VERIFIED-IMPROVEMENT":
        from .reports import verify_selected_profile
        verify_selected_profile(profile)
    if not profile.get("model", {}).get("files") or profile["model"].get("path") != profile["model"]["files"][0]["path"]:
        raise ValueError("Model primary path differs from registered model files; run setup again.")
    for artifact in profile["model"]["files"]:
        path = Path(artifact["path"])
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or file_digest(path, deadline=deadline) != artifact["sha256"]:
            raise ValueError("Model file changed or is missing; create a new profile with local setup.")
    verify_runtime(profile["runtime"], deadline=deadline)
