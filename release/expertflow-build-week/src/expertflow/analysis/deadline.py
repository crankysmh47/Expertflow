"""Backend-specific deadline bounds over measured router timestamps."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics

from expertflow.trace.schema import RouterTraceEvent


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def evaluate_one_layer_oracle(
    training_events: list[RouterTraceEvent] | tuple[RouterTraceEvent, ...],
    evaluation_traces: list[
        list[RouterTraceEvent] | tuple[RouterTraceEvent, ...]
    ],
    *,
    capacity_per_layer: int,
    expert_transfer_ms: float,
    transfer_backend: str | None = None,
    window_backend: str | None = None,
    transfer_statistic: str | None = None,
) -> dict[str, object]:
    """Evaluate a frozen cache and an oracle one-layer issue boundary."""

    training = tuple(training_events)
    traces = [tuple(trace) for trace in evaluation_traces]
    if not training or not traces or any(not trace for trace in traces):
        raise ValueError("training and evaluation traces must not be empty")
    if capacity_per_layer <= 0 or expert_transfer_ms <= 0:
        raise ValueError("capacity and transfer time must be positive")
    if (transfer_backend is None) != (window_backend is None):
        raise ValueError("transfer_backend and window_backend must both be declared")
    if transfer_statistic is not None and transfer_backend is None:
        raise ValueError("transfer_statistic requires declared timing backends")
    if transfer_backend is not None and (
        not transfer_backend.strip()
        or not window_backend
        or not window_backend.strip()
        or transfer_statistic is None
        or not transfer_statistic.strip()
    ):
        raise ValueError("declared timing evidence labels must be non-empty")
    if any(
        len(event.selected_expert_ids) > capacity_per_layer
        for event in training
    ):
        raise ValueError("capacity cannot be below router top-k")

    counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for event in training:
        counts[event.layer_id].update(event.selected_expert_ids)
    residents = {
        layer_id: frozenset(
            expert_id
            for expert_id, _ in sorted(
                layer_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:capacity_per_layer]
        )
        for layer_id, layer_counts in counts.items()
    }
    expected_layers = frozenset(residents)

    token_count = 0
    demand_count = 0
    hit_count = 0
    no_prefetch_blocking_ms = 0.0
    oracle_residual_ms = 0.0
    on_time_events = 0
    late_events = 0
    windows: list[float] = []
    timeline: list[dict[str, object]] = []
    for source_index, trace in enumerate(traces):
        groups: defaultdict[tuple[str, int], list[RouterTraceEvent]] = (
            defaultdict(list)
        )
        for event in trace:
            if len(event.selected_expert_ids) > capacity_per_layer:
                raise ValueError("capacity cannot be below router top-k")
            groups[(event.request_id, event.forward_id)].append(event)
        for events in groups.values():
            events.sort(key=lambda event: event.layer_id)
            if frozenset(event.layer_id for event in events) != expected_layers:
                raise ValueError(
                    "every evaluation token must cover the training layer set"
                )
            token_count += 1
            previous: RouterTraceEvent | None = None
            for event in events:
                missing = tuple(
                    expert_id
                    for expert_id in event.selected_expert_ids
                    if expert_id not in residents[event.layer_id]
                )
                hits = len(event.selected_expert_ids) - len(missing)
                demand_count += len(event.selected_expert_ids)
                hit_count += hits
                transfer_ms = len(missing) * expert_transfer_ms
                no_prefetch_blocking_ms += transfer_ms
                available_window_ms: float | None = None
                if previous is None:
                    residual_ms = transfer_ms
                else:
                    available_window_ms = (
                        event.observed_at_ns - previous.observed_at_ns
                    ) / 1_000_000
                    if available_window_ms < 0:
                        raise ValueError(
                            "evaluation timestamps must be monotonic"
                        )
                    windows.append(available_window_ms)
                    residual_ms = max(
                        0.0, transfer_ms - available_window_ms
                    )
                if residual_ms > 0:
                    late_events += 1
                else:
                    on_time_events += 1
                oracle_residual_ms += residual_ms
                timeline.append(
                    {
                        "source_index": source_index,
                        "request_id": event.request_id,
                        "forward_id": event.forward_id,
                        "token_index": event.token_index,
                        "layer_id": event.layer_id,
                        "missing_expert_ids": list(missing),
                        "transfer_ms": transfer_ms,
                        "available_window_ms": available_window_ms,
                        "oracle_residual_blocking_ms": residual_ms,
                        "oracle_status": (
                            "late" if residual_ms > 0 else "ready"
                        ),
                    }
                )
                previous = event

    if not windows:
        raise ValueError("evaluation requires at least two layers per token")
    measurement_kind = "estimated"
    timing_evidence: dict[str, object] | None = None
    if transfer_backend is not None and window_backend is not None:
        measurement_kind = (
            "estimated_cross_backend"
            if transfer_backend != window_backend
            else "estimated_backend_specific"
        )
        timing_evidence = {
            "transfer_backend": transfer_backend,
            "window_backend": window_backend,
            "transfer_statistic": transfer_statistic,
            "contention_measured": False,
            "live_runtime_measurement": False,
        }
    report = {
        "schema_version": "1.0.0",
        "measurement_kind": measurement_kind,
        "window_measurement_kind": "measured_backend_specific",
        "fit_scope": "held_out",
        "capacity_per_layer": capacity_per_layer,
        "expert_transfer_ms": expert_transfer_ms,
        "training_event_count": len(training),
        "token_count": token_count,
        "event_count": len(timeline),
        "expert_demand_count": demand_count,
        "static_hit_rate": hit_count / demand_count,
        "blocking_no_prefetch_ms_per_token": (
            no_prefetch_blocking_ms / token_count
        ),
        "observed_adjacent_layer_window_ms": {
            "sample_count": len(windows),
            "min": min(windows),
            "median": statistics.median(windows),
            "mean": statistics.fmean(windows),
            "p95": _percentile(windows, 0.95),
            "max": max(windows),
        },
        "one_layer_oracle": {
            "on_time_event_count": on_time_events,
            "late_event_count": late_events,
            "residual_blocking_ms_per_token": (
                oracle_residual_ms / token_count
            ),
        },
        "timeline": timeline,
    }
    if timing_evidence is not None:
        report["timing_evidence"] = timing_evidence
    return report
