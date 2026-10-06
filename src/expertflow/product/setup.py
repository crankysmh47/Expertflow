"""Register a local model/runtime and independently verify baseline loading."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import uuid
from expertflow.doctor import collect_doctor_report
from .models import inspect_model
from .profiles import create_profile, save_profile
from .runtime import inspect_runtime
from .server import ServerSession


def setup_profile(model_path: Path, runtime_dir: Path, output: Path, *, context=4096,
                  gpu_layers=None, threads=None, cpu_moe=False, dll_dirs=(),
                  name=None, probe=True, health_timeout=180) -> dict:
    model = inspect_model(model_path)
    runtime = inspect_runtime(runtime_dir, dll_dirs=dll_dirs)
    profile = create_profile(model, runtime, context=context, gpu_layers=gpu_layers,
                             threads=threads, cpu_moe=cpu_moe, name=name)
    hardware = collect_doctor_report(model_path.resolve().parent)
    profile["host"]["gpus"] = [{k: gpu[k] for k in ["index", "name", "driver_version", "memory_total_mib"]} for gpu in hardware["gpus"]]
    profile["host"]["system_ram_bytes"] = hardware["system_ram_bytes"]
    profile["status"] = "UNVERIFIED-PROFILE"
    log_dir = output.resolve().parent / "logs" / (profile["id"] + "-" + uuid.uuid4().hex)
    profile["log_directory"] = str(log_dir)
    save_profile(output, profile)
    if not probe:
        return profile
    try:
        with ServerSession(profile, log_dir=log_dir, health_timeout=health_timeout) as server:
            result = server.complete("Say hello in one short sentence.", predict=8)
            count = result.get("tokens_predicted", result.get("timings", {}).get("predicted_n", 0))
            if not isinstance(count, int) or count < 1:
                raise ValueError("Load probe returned no generated tokens; baseline is unverified.")
            profile["load_receipt"] = {"created_at": datetime.now(timezone.utc).isoformat(),
                "tokens_predicted": count, "tokens": result.get("tokens"),
                "timings": result.get("timings"), "wall_seconds": result["wall_seconds"],
                "runtime_props": server.props, "prompt": "Say hello in one short sentence.",
                "predict": 8, "scope": "Load/generation smoke only; no tuning or serving-performance acceptance."}
        profile.update(load_verified=True, status="RUNNABLE-UNTUNED")
        save_profile(output, profile)
        return profile
    except (ValueError, OSError) as error:
        profile.update(load_verified=False, status="ENVIRONMENT-BLOCKED", load_error=str(error))
        save_profile(output, profile)
        raise
