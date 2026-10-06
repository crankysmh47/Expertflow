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
    capacity_curve: Mapping[str, object] | None = None,
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
    expert_cache: dict[str, object] | None = None
    provenance = {
        "hardware": "measured",
        "baseline": "measured",
        "locality": "measured",
        "policy": "estimated",
    }
    if capacity_curve is not None:
        _require_evidence_kind(
            capacity_curve, "capacity_curve", "estimated"
        )
        fit_scope = capacity_curve.get("fit_scope")
        if fit_scope not in {"in_sample", "held_out"}:
            raise RecommendationInputError(
                "capacity_curve fit_scope must be 'in_sample' or 'held_out'"
            )
        slot_bytes = _integer(
            capacity_curve.get("slot_bytes"), "capacity_curve.slot_bytes"
        )
        if slot_bytes <= 0:
            raise RecommendationInputError(
                "capacity_curve.slot_bytes must be positive"
            )
        expert_transfer_ms = _number(
            capacity_curve.get("expert_transfer_ms"),
            "capacity_curve.expert_transfer_ms",
        )
        if expert_transfer_ms <= 0:
            raise RecommendationInputError(
                "capacity_curve.expert_transfer_ms must be positive"
            )
        curve_layers = capacity_curve.get("layer_ids")
        if not isinstance(curve_layers, list) or not curve_layers:
            raise RecommendationInputError(
                "capacity_curve.layer_ids must contain layers"
            )
        if any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in curve_layers
        ):
            raise RecommendationInputError(
                "capacity_curve.layer_ids must contain integers"
            )
        points = capacity_curve.get("points")
        if not isinstance(points, list) or not points:
            raise RecommendationInputError(
                "capacity_curve.points must contain points"
            )

        eligible: list[tuple[int, int, Mapping[str, object]]] = []
        headroom_bytes = headroom_mib * 1024 * 1024
        for index, value in enumerate(points):
            point = _mapping(value, f"capacity_curve.points[{index}]")
            point_capacity = _integer(
                point.get("capacity_per_layer"),
                f"capacity_curve.points[{index}].capacity_per_layer",
            )
            cache_bytes = _integer(
                point.get("projected_cache_bytes"),
                f"capacity_curve.points[{index}].projected_cache_bytes",
            )
            if point_capacity <= 0 or cache_bytes <= 0:
                raise RecommendationInputError(
                    "capacity_curve point capacity and bytes must be positive"
                )
            if cache_bytes <= headroom_bytes:
                eligible.append((point_capacity, cache_bytes, point))
        if not eligible:
            reason_codes = ["NO_CAPACITY_CURVE_POINT_FITS_HEADROOM"]
        else:
            curve_capacity, cache_bytes, point = max(
                eligible, key=lambda item: item[0]
            )
            curve_policies: list[tuple[str, float, float]] = []
            for name in ("static_hotset", "lru"):
                policy = _mapping(
                    point.get(name),
                    f"capacity_curve.capacity_{curve_capacity}.{name}",
                )
                hit_rate = _number(
                    policy.get("hit_rate"),
                    f"capacity_curve.capacity_{curve_capacity}.{name}.hit_rate",
                )
                transfer_cost = _number(
                    policy.get(
                        "estimated_serial_h2d_ms_per_layer_sweep"
                    ),
                    "capacity_curve policy transfer estimate",
                )
                if not 0 <= hit_rate <= 1 or transfer_cost < 0:
                    raise RecommendationInputError(
                        "capacity_curve policy values are out of range"
                    )
                curve_policies.append((name, hit_rate, transfer_cost))
            selected_policy, selected_hit_rate, selected_transfer_ms = max(
                curve_policies, key=lambda item: (item[1], item[0])
            )
            lru_hit_rate = next(
                hit_rate
                for name, hit_rate, _ in curve_policies
                if name == "lru"
            )
            capacity = curve_capacity
            total_events = _integer(
                capacity_curve.get("event_count"),
                "capacity_curve.event_count",
            )
            expert_cache = {
                "fit_scope": fit_scope,
                "target_layer_count": len(curve_layers),
                "slot_bytes": slot_bytes,
                "capacity_per_layer": curve_capacity,
                "projected_cache_bytes": cache_bytes,
                "projected_cache_mib": cache_bytes / (1024 * 1024),
                "remaining_headroom_after_cache_mib": (
                    (headroom_bytes - cache_bytes) / (1024 * 1024)
                ),
                "measured_expert_transfer_ms": expert_transfer_ms,
                "estimated_serial_h2d_ms_per_layer_sweep": (
                    selected_transfer_ms
                ),
            }
            reason_codes = [
                "PER_LAYER_DEADLINES_NOT_MEASURED",
                "END_TO_END_CACHE_NOT_MEASURED",
            ]
            if fit_scope != "held_out":
                reason_codes.append("HELD_OUT_POLICY_REQUIRED")
            provenance.update(
                {
                    "expert_layout": "source_derived",
                    "transfer": "measured",
                    "stratified_workload": "measured",
                }
            )
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
        "provenance": provenance,
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
            "layer_count": (
                expert_cache["target_layer_count"]
                if expert_cache is not None
                else len(layers)
            ),
        },
        "reason_codes": reason_codes,
        **({"expert_cache": expert_cache} if expert_cache is not None else {}),
    }
