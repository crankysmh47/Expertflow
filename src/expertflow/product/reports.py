"""Reconstruct local receipts without native calls; export only safe summaries."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import statistics
import zipfile
from .runtime import file_digest


def read_json(path: Path):
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ValueError(f"Missing/oversized receipt: {path.name}.")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Expected an object receipt.")
        return result
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid receipt: {path.name}.") from error


def freeze_sources(output: Path) -> dict:
    # Includes the lower-level process, HTTP, telemetry and host helpers, rather
    # than pretending product/ alone is the complete execution dependency.
    root = Path(__file__).resolve().parents[1]
    sources = sorted([*root.rglob("*.py"), root / "product" / "stock-policy.json"])
    hashes = {}
    with zipfile.ZipFile(output / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            name = source.relative_to(root).as_posix()
            payload = source.read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
    return hashes


def _measurement(directory: Path, prompts: list, repeats: int, predict: int) -> dict:
    from .benchmark import parse_completion
    saved = read_json(directory / "measurement.json")
    for name, digest in saved.get("artifact_sha256", {}).items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe measurement receipt path.")
        target = directory / name
        if target.is_symlink() or not target.is_file() or file_digest(target) != digest:
            raise ValueError(f"Measurement artifact changed/missing: {name}.")
    if saved.get("status") != "MEASURED" or saved.get("cleanup") is not True:
        raise ValueError("Measurement is incomplete or cleanup is unverified.")
    records = []
    for repetition in range(repeats):
        for index, prompt in enumerate(prompts):
            records.append(parse_completion(read_json(directory / f"response-{repetition}-{index}.json"), prompt, expected_predict=predict))
    if records != saved.get("records"):
        raise ValueError("Measurement records differ from raw responses.")
    checks = {"decode_tps": statistics.mean(r["decode_tps"] for r in records),
              "tokens": [t for r in records for t in r["tokens"]],
              "prompt_tokens": [r["prompt_tokens"] for r in records]}
    if any(saved.get(k) != v for k, v in checks.items()):
        raise ValueError("Measurement summary differs from raw responses.")
    receipt = read_json(directory / "process-receipt.json")
    if receipt.get("cleanup") is not True:
        raise ValueError("Owned process cleanup receipt is missing.")
    return saved


def verify_job(directory: Path) -> dict:
    """Detect missing/changed evidence and reconstruct the fixed decision."""
    directory = directory.resolve()
    try:
        registration = read_json(directory / "registration.json")
        report = read_json(directory / "report.json")
        if file_digest(directory / "registration.json") != report.get("registration_sha256"):
            raise ValueError("Job registration changed since collection.")
        count = len(list(directory.glob("*/launch.json")))
        if count > registration["maximum_model_processes"]:
            raise ValueError("Declared process budget exceeded.")
        if report["status"] in {"INCONCLUSIVE", "UNSUPPORTED"}:
            return {"status":"PARTIAL", "archived_status":report["status"], "native_calls":0,
                    "launched_model_processes":count, "reason":"Partial diagnostics retained; no acceptance claim."}
        if not (directory / "source.zip").is_file():
            raise ValueError("Missing frozen source receipt.")
        with zipfile.ZipFile(directory / "source.zip") as archive:
            for name, digest in registration["sources"].items():
                archived_name = name if name in archive.namelist() else "src/expertflow/product/" + name
                if hashlib.sha256(archive.read(archived_name)).hexdigest() != digest:
                    raise ValueError("Frozen source receipt differs from registration.")
        if registration.get("kind") == "local-benchmark":
            measurement = _measurement(directory / "measurement", registration["prompts"], registration["repeats"], registration["predict"])
            if measurement != report["measurement"] or count != 1 or report["status"] != "MEASURED":
                raise ValueError("Benchmark report differs from collected receipt.")
        else:
            from .tuning import decide_pairs
            search = report["search"]
            if not search:
                raise ValueError("No baseline search receipt.")
            host = search[0]["measurement"]["host_environment"]
            for index, trial in enumerate(search):
                name = "search-baseline" if index == 0 else f"search-{trial['threads']}"
                measured = _measurement(directory / name, [registration["train_prompt"]], registration["search_repeats"], registration["predict"])
                if measured != trial["measurement"] or measured["host_environment"] != host:
                    raise ValueError("Search receipt/host changed.")
                if measured["tokens"] != search[0]["measurement"]["tokens"] or measured["prompt_tokens"] != search[0]["measurement"]["prompt_tokens"]:
                    raise ValueError("Search exact token guard failed.")
            pairs = report["pairs"]
            for index, pair in enumerate(pairs):
                for role in ["baseline", "candidate"]:
                    measured = _measurement(directory / f"confirm-{index}-{role}", registration["heldout_prompts"], 1, registration["predict"])
                    if measured != pair[role] or measured["host_environment"] != host:
                        raise ValueError("Confirmation receipt/host changed.")
            if pairs:
                decision = decide_pairs(pairs)
                if any(report.get(k) != v for k, v in decision.items()):
                    raise ValueError("Tuning decision differs from reconstructed pairs.")
            elif report["status"] != "NO-MEASURABLE-GAIN" or max(t["measurement"]["decode_tps"] for t in search) / search[0]["measurement"]["decode_tps"] >= 1.05:
                raise ValueError("No-gain result does not match search receipts.")
            if count != len(search) + 2 * len(pairs):
                raise ValueError("Collected process count differs from receipts.")
        return {"status":"VERIFIED-RECEIPTS", "archived_status":report["status"], "native_calls":0,
                "launched_model_processes":count, "reason":"Reconstructed archived receipts; not a fresh performance or quality claim."}
    except (KeyError, TypeError, IndexError, OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Cannot reconstruct job receipts: {error}") from error


def verify_selected_profile(profile: dict) -> None:
    evidence = profile.get("evidence", [])
    if not evidence or not isinstance(evidence[-1], dict):
        raise ValueError("Verified improvement requires its original evidence; use the baseline profile.")
    receipt = evidence[-1]
    path = Path(receipt.get("report", ""))
    if not path.is_file() or file_digest(path) != receipt.get("report_sha256"):
        raise ValueError("Selected-profile evidence changed/missing; use the baseline or requalify.")
    verify_job(path.parent)
    report, registration = read_json(path), read_json(path.parent / "registration.json")
    if report["status"] != "VERIFIED-IMPROVEMENT" or report.get("selected_settings") != profile["settings"]:
        raise ValueError("Selected settings do not match accepted evidence.")
    for key in ["model", "runtime", "host"]:
        if profile[key] != registration["profile"][key]:
            raise ValueError(f"Selected profile {key} differs from evidence.")
    from .benchmark import capture_performance_host
    if capture_performance_host(profile) != report["search"][0]["measurement"]["host_environment"]:
        raise ValueError("Selected evidence host/driver changed; use the baseline or requalify.")


def support_summary(directory: Path) -> dict:
    registration, report = read_json(directory / "registration.json"), read_json(directory / "report.json")
    # Explicit allowlist: never recursively copy logs, prompts, names, commands,
    # raw exception strings, paths or arbitrary user-provided profile fields.
    return {"schema_version":1, "kind":"redacted-local-support", "status":report.get("status"),
            "model_processes":report.get("model_processes"), "wall_seconds":report.get("wall_seconds"),
            "registration_sha256":file_digest(directory / "registration.json"),
            "report_sha256":file_digest(directory / "report.json"),
            "context":registration.get("profile", {}).get("settings", {}).get("context"),
            "privacy":"No prompts, responses, logs, names, local paths or environment variables included."}
