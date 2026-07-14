"""Evidence-bounded machine-specific recommendations."""

from __future__ import annotations

from collections.abc import Mapping


class RecommendationInputError(ValueError):
    """Raised when recommendation evidence is missing or mislabeled."""


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecommendationInputError(f"{field} must be an object")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecommendationInputError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecommendationInputError(f"{field} must be a number")
    return float(value)


def _require_evidence_kind(
    document: Mapping[str, object],
    name: str,
    expected: str,
) -> None:
    if document.get("schema_version") != "1.0.0":
        raise RecommendationInputError(f"{name} schema_version must be '1.0.0'")
    if document.get("measurement_kind") != expected:
        raise RecommendationInputError(
            f"{name} measurement_kind must be {expected!r}"
        )


def build_recommendation(
    doctor: Mapping[str, object],
    baseline: Mapping[str, object],
    profile: Mapping[str, object],
    simulation: Mapping[str, object],
    *,
    safety_reserve_mib: int = 1_024,
) -> dict[str, object]:
    """Build a conservative recommendation from measured and estimated evidence."""

    if safety_reserve_mib < 0:
        raise RecommendationInputError("safety_reserve_mib must not be negative")
    _require_evidence_kind(doctor, "doctor", "measured")
    _require_evidence_kind(baseline, "baseline", "measured")
    _require_evidence_kind(profile, "profile", "measured")
    _require_evidence_kind(simulation, "simulation", "estimated")
    if baseline.get("status") != "passed":
        raise RecommendationInputError("baseline status must be 'passed'")

    gpus = doctor.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise RecommendationInputError("doctor gpus must contain at least one GPU")
    gpu = _mapping(gpus[0], "doctor.gpus[0]")
    gpu_index = _integer(gpu.get("index"), "doctor.gpus[0].index")
    total_mib = _integer(
        gpu.get("memory_total_mib"), "doctor.gpus[0].memory_total_mib"
    )
    gpu_name = gpu.get("name")
    if not isinstance(gpu_name, str) or not gpu_name:
        raise RecommendationInputError("doctor.gpus[0].name must be a string")

    memory = _mapping(baseline.get("memory"), "baseline.memory")
    peak_by_gpu = _mapping(
        memory.get("peak_gpu_used_mib"), "baseline.memory.peak_gpu_used_mib"
    )
    peak_mib = _integer(
        peak_by_gpu.get(str(gpu_index)),
        f"baseline.memory.peak_gpu_used_mib.{gpu_index}",
    )
    headroom_mib = max(0, total_mib - peak_mib - safety_reserve_mib)

    profile_body = _mapping(profile.get("profile"), "profile.profile")
    total_events = _integer(
        profile_body.get("total_events"), "profile.profile.total_events"
    )
    layers = profile_body.get("layers")
    if not isinstance(layers, list):
        raise RecommendationInputError("profile.profile.layers must be an array")

    simulation_body = _mapping(
        simulation.get("simulation"), "simulation.simulation"
    )
    capacity = _integer(
        simulation_body.get("capacity_per_layer"),
        "simulation.simulation.capacity_per_layer",
    )
    policies: list[tuple[str, float]] = []
    for name in ("static_hotset", "lru"):
        result = _mapping(
            simulation_body.get(name), f"simulation.simulation.{name}"
        )
        policies.append(
            (
                name,
                _number(
                    result.get("hit_rate"),
                    f"simulation.simulation.{name}.hit_rate",
                ),
            )
        )
    selected_policy, selected_hit_rate = max(
        policies, key=lambda item: (item[1], item[0])
    )
    lru_hit_rate = dict(policies)["lru"]

    reason_codes = [
        "EXPERT_BYTES_NOT_MEASURED",
        "TRANSFER_TIMING_NOT_MEASURED",
        "STRATIFIED_TRACE_REQUIRED",
    ]
    if selected_policy == "static_hotset" and selected_hit_rate > lru_hit_rate:
        reason_codes.append("STATIC_HOTSET_OUTPERFORMS_LRU")
    verdict = "CONDITIONAL"
    if headroom_mib == 0:
        verdict = "DO_NOT_ENABLE"
        reason_codes.append("INSUFFICIENT_MEASURED_HEADROOM")

    return {
        "schema_version": "1.0.0",
        "verdict": verdict,
        "live_cache_enabled": False,
        "provenance": {
            "hardware": "measured",
            "baseline": "measured",
            "locality": "measured",
            "policy": "estimated",
        },
        "hardware": {
            "gpu_index": gpu_index,
            "gpu_name": gpu_name,
            "total_vram_mib": total_mib,
            "measured_peak_vram_mib": peak_mib,
            "safety_reserve_mib": safety_reserve_mib,
            "remaining_configurable_headroom_mib": headroom_mib,
        },
        "replay": {
            "policy": selected_policy,
            "capacity_per_layer": capacity,
            "estimated_hit_rate": selected_hit_rate,
            "estimated_lru_hit_rate": lru_hit_rate,
        },
        "trace": {
            "event_count": total_events,
            "layer_count": len(layers),
        },
        "reason_codes": reason_codes,
    }
