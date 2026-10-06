"""Fixed-budget thread search with separate held-out paired confirmation."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from .benchmark import measure_profile, write_json, TRAIN_PROMPT, HELDOUT_PROMPTS
from .qualification import eligible_threads
from .runtime import file_digest


def decide_pairs(pairs: list[dict]) -> dict:
    if len(pairs) != 5:
        return {"status": "INCONCLUSIVE", "reason": "Five complete independent confirmation pairs are required."}
    changes = []
    for pair in pairs:
        baseline, candidate = pair["baseline"], pair["candidate"]
        if baseline["tokens"] != candidate["tokens"] or baseline.get("prompt_tokens") != candidate.get("prompt_tokens"):
            return {"status": "INCONCLUSIVE", "reason": "Exact token/prompt identity mismatch; baseline retained."}
        a, b = baseline["decode_tps"], candidate["decode_tps"]
        if any(type(t) not in {int, float} or not math.isfinite(t) or t <= 0 for t in [a, b]):
            return {"status": "INCONCLUSIVE", "reason": "Invalid paired timings; baseline retained."}
        changes.append(100 * (b / a - 1))
    rng = random.Random(20261006)
    bootstrap = sorted(statistics.mean(rng.choices(changes, k=5)) for _ in range(10000))
    ci = [bootstrap[249], bootstrap[9749]]
    mean = statistics.mean(changes)
    noisy = statistics.stdev(changes) > 10
    status = "INCONCLUSIVE" if noisy else "VERIFIED-IMPROVEMENT" if ci[0] > 5 else "NO-MEASURABLE-GAIN"
    return {"status": status, "gain_pct": mean, "ci95_pct": ci,
            "reason": "Paired variance exceeds the fixed 10 percentage-point limit." if noisy else "Held-out lower 95% gain bound exceeds 5%." if status == "VERIFIED-IMPROVEMENT" else "Held-out confirmation did not clear the fixed 5% practical gain gate; baseline retained."}


def tune_profile(profile: dict, output: Path, *, budget_seconds=600) -> dict:
    if type(budget_seconds) not in {int, float} or not math.isfinite(budget_seconds) or not 1 <= budget_seconds <= 86400:
        raise ValueError("Tuning budget must be 1–86400 seconds.")
    if output.exists():
        raise ValueError("Job output already exists; choose a fresh directory. Completed budgets cannot be reused.")
    output.mkdir(parents=True)
    explicit_threads = profile["settings"].get("threads") is not None
    candidates = [] if explicit_threads else eligible_threads(profile)
    from .reports import freeze_sources
    registration = {"schema_version": 1, "profile": profile,
        "sources": freeze_sources(output),
        "candidate_threads": candidates, "baseline_threads": profile["settings"].get("threads"),
        "search_repeats": 3, "predict": 256, "train_prompt": TRAIN_PROMPT,
        "heldout_prompts": HELDOUT_PROMPTS, "confirmation_pairs": 5,
        "minimum_gain_lower_ci95_pct": 5, "maximum_pair_sd_pct": 10,
        "interprocess_spacing_seconds": 30, "budget_seconds": budget_seconds,
        "maximum_external_gpu_utilization_pct": 5,
        "idle_settle_seconds":5, "consecutive_idle_samples":3,
        "maximum_model_processes": 1 + len(candidates) + 10,
        "policy": "exact", "resume": False}
    write_json(output / "registration.json", registration)
    result = {"status": "INCONCLUSIVE", "model_processes": 0, "search": [], "pairs": [],
              "baseline_retained": True, "registration_sha256": file_digest(output / "registration.json")}
    start = time.monotonic()
    deadline = start + budget_seconds
    last_end = None
    winner = None
    def measure(p, name, **kwargs):
        nonlocal last_end
        if last_end is not None:
            wait = max(0, 30 - (time.monotonic() - last_end))
            if time.monotonic() + wait >= deadline:
                raise TimeoutError("Wall budget cannot fund next paced measurement; baseline retained.")
            time.sleep(wait)
        if time.monotonic() >= deadline:
            raise TimeoutError("Wall budget exhausted; baseline retained.")
        result["model_processes"] += 1
        write_json(output / "journal.json", {"attempts": result["model_processes"], "next_measurement": name,
                                          "maximum": registration["maximum_model_processes"]})
        try:
            return measure_profile(p, output / name, deadline=deadline, **kwargs)
        finally:
            last_end = time.monotonic()
    try:
        if not candidates:
            reason = "Exact tuning requires upstream default threads. Create a new setup profile without --threads; your existing profile is unchanged." if explicit_threads else "No reviewed exact scheduling policy matches this model/runtime/placement. Run or benchmark the untuned baseline; no optimization processes were launched."
            result.update(status="UNSUPPORTED", reason=reason)
            return result
        baseline = measure(profile, "search-baseline")
        result["search"].append({"threads": profile["settings"].get("threads"), "measurement": baseline})
        best, winner = baseline["decode_tps"], None
        for threads in candidates:
            if threads == profile["settings"].get("threads"):
                continue
            trial = copy.deepcopy(profile)
            trial["settings"]["threads"] = threads
            measurement = measure(trial, f"search-{threads}")
            if measurement.get("host_environment") != baseline.get("host_environment"):
                raise ValueError("Host controls changed between measurements; baseline retained.")
            if measurement["tokens"] != baseline["tokens"] or measurement["prompt_tokens"] != baseline["prompt_tokens"]:
                raise ValueError("Search exact tokens differ; baseline retained.")
            result["search"].append({"threads": threads, "measurement": measurement})
            if measurement["decode_tps"] > best:
                best, winner = measurement["decode_tps"], trial
        if winner is None or best / baseline["decode_tps"] < 1.05:
            result.update(status="NO-MEASURABLE-GAIN", reason="Search found no candidate with at least 5% apparent gain; no confirmation launched.")
            return result
        for index in range(5):
            order = [("baseline", profile), ("candidate", winner)] if index % 2 == 0 else [("candidate", winner), ("baseline", profile)]
            pair = {}
            for role, p in order:
                pair[role] = measure(p, f"confirm-{index}-{role}", prompts=HELDOUT_PROMPTS, repeats=1)
                if pair[role].get("host_environment") != baseline.get("host_environment"):
                    raise ValueError("Host controls changed during confirmation; baseline retained.")
            result["pairs"].append(pair)
        result.update(decide_pairs(result["pairs"]))
        if result["status"] == "VERIFIED-IMPROVEMENT":
            winner.update(status="VERIFIED-IMPROVEMENT")
            result.update(baseline_retained=False, selected_settings=winner["settings"])
            winner["evidence"].append({"report": str(output.resolve() / "report.json"), "registration_sha256": result["registration_sha256"],
                "scope": "Fixed model/runtime/host/context/raw workloads; threads only, not serving performance or a global optimum."})
            savings = statistics.mean(sum(r["predicted_tokens"] / r["decode_tps"] for r in pair["baseline"]["records"]) - sum(r["predicted_tokens"] / r["decode_tps"] for r in pair["candidate"]["records"]) for pair in result["pairs"])
            result["break_even"] = {"scope":"Same two held-out decode workloads; excludes prompt processing, model load and serving.",
                                    "saved_decode_seconds_per_workload":savings}
        return result
    except (ValueError, OSError, RuntimeError, TimeoutError, KeyboardInterrupt) as error:
        result.update(status="INCONCLUSIVE", reason=str(error) or "Cancelled; partial evidence retained and baseline unchanged.")
        return result
    finally:
        result["model_attempts"] = result["model_processes"]
        result["model_processes"] = len(list(output.glob("*/launch.json")))
        result["wall_seconds"] = time.monotonic() - start
        if result.get("break_even"):
            savings = result["break_even"]["saved_decode_seconds_per_workload"]
            result["break_even"]["comparable_workloads"] = math.ceil(result["wall_seconds"] / savings) if savings > 0 else None
        write_json(output / "report.json", result)
        if result["status"] == "VERIFIED-IMPROVEMENT" and winner is not None:
            winner["evidence"][-1]["report_sha256"] = file_digest(output / "report.json")
            write_json(output / "selected-profile.json", winner)
