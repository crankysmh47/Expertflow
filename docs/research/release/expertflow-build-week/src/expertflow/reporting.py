"""Standalone, escaped Observatory replay reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path

from expertflow.analysis.replay import PolicyReplay


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _items(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an array")
    return value


def _physical_evidence_sections(
    evidence: Mapping[str, object], *, live_cache_enabled: object
) -> str:
    if live_cache_enabled is not False:
        raise ValueError(
            "physical evidence report requires live_cache_enabled=false"
        )
    breakdown = _mapping(
        evidence.get("heldout_breakdown"), "heldout_breakdown"
    )
    layout = _mapping(evidence.get("expert_layout"), "expert_layout")
    transfer = _mapping(evidence.get("transfer"), "transfer")
    deadline = _mapping(evidence.get("deadline"), "deadline")
    sources = _mapping(evidence.get("sources"), "sources")

    capacity = _integer(
        breakdown.get("capacity_per_layer"), "capacity_per_layer"
    )
    slot_bytes = _integer(breakdown.get("slot_bytes"), "slot_bytes")
    layers = _items(breakdown.get("layer_ids"), "layer_ids")
    if not layers or not all(
        isinstance(layer, int) and not isinstance(layer, bool)
        for layer in layers
    ):
        raise ValueError("layer_ids must contain integers")
    if breakdown.get("lru_reset_scope") != "conversation":
        raise ValueError("held-out LRU must reset per conversation")
    aggregate = _mapping(breakdown.get("aggregate"), "aggregate")
    static = _mapping(aggregate.get("static_hotset"), "static_hotset")
    lru = _mapping(aggregate.get("lru"), "lru")
    static_rate = _number(static.get("hit_rate"), "static_hotset.hit_rate")
    lru_rate = _number(lru.get("hit_rate"), "lru.hit_rate")
    static_misses = _integer(
        static.get("miss_count"), "static_hotset.miss_count"
    )
    lru_misses = _integer(lru.get("miss_count"), "lru.miss_count")
    static_cold_bytes = _integer(
        static.get("cold_bytes"), "static_hotset.cold_bytes"
    )
    lru_cold_bytes = _integer(lru.get("cold_bytes"), "lru.cold_bytes")
    cold_reduction = (
        (lru_cold_bytes - static_cold_bytes) / lru_cold_bytes
        if lru_cold_bytes
        else 0.0
    )

    projection = _mapping(layout.get("projection"), "projection")
    projected_capacity = _integer(
        projection.get("capacity_per_layer"), "projection.capacity_per_layer"
    )
    target_layer_count = _integer(
        projection.get("target_layer_count"),
        "projection.target_layer_count",
    )
    slot_count = _integer(projection.get("slot_count"), "projection.slot_count")
    projected_cache_bytes = _integer(
        layout.get("projected_cache_bytes"), "projected_cache_bytes"
    )
    projected_slot_min = _integer(
        layout.get("projected_slot_bytes_min"),
        "projected_slot_bytes_min",
    )
    projected_slot_max = _integer(
        layout.get("projected_slot_bytes_max"),
        "projected_slot_bytes_max",
    )
    expected_cache_bytes = capacity * len(layers) * slot_bytes
    if not (
        projected_capacity == capacity
        and target_layer_count == len(layers)
        and slot_count == capacity * len(layers)
        and projected_slot_min == projected_slot_max == slot_bytes
        and projected_cache_bytes == expected_cache_bytes
    ):
        raise ValueError("packed expert layout does not match static cache budget")
    encoded_min = _integer(
        layout.get("encoded_object_bytes_min"), "encoded_object_bytes_min"
    )
    encoded_max = _integer(
        layout.get("encoded_object_bytes_max"), "encoded_object_bytes_max"
    )
    object_count = _integer(layout.get("object_count"), "object_count")
    alignment = _integer(layout.get("alignment_bytes"), "alignment_bytes")

    runs = _items(transfer.get("runs"), "transfer.runs")

    def transfer_run(memory: str) -> Mapping[str, object]:
        for raw_run in runs:
            run = _mapping(raw_run, "transfer.runs[]")
            if (
                run.get("direction") == "host_to_device"
                and run.get("source_memory") == memory
                and run.get("payload_bytes") == slot_bytes
            ):
                return run
        raise ValueError(
            f"transfer evidence has no {memory} expert-slot H2D run"
        )

    pinned = transfer_run("pinned")
    pageable = transfer_run("pageable")
    pinned_single = _mapping(
        pinned.get("single_copy_cuda_event"), "pinned.single_copy_cuda_event"
    )
    pageable_single = _mapping(
        pageable.get("single_copy_cuda_event"),
        "pageable.single_copy_cuda_event",
    )
    sustained = _mapping(
        pinned.get("cuda_event_per_copy"), "pinned.cuda_event_per_copy"
    )
    enqueue = _mapping(pinned.get("host_enqueue"), "pinned.host_enqueue")
    pinned_p50 = _number(pinned_single.get("p50_ms"), "pinned.p50_ms")
    pinned_p95 = _number(pinned_single.get("p95_ms"), "pinned.p95_ms")
    pageable_p50 = _number(
        pageable_single.get("p50_ms"), "pageable.p50_ms"
    )
    pageable_p95 = _number(
        pageable_single.get("p95_ms"), "pageable.p95_ms"
    )
    sustained_gib_s = _number(
        sustained.get("mean_gib_per_second"), "mean_gib_per_second"
    )
    enqueue_p50 = _number(enqueue.get("p50_ms"), "host_enqueue.p50_ms")
    enqueue_p95 = _number(enqueue.get("p95_ms"), "host_enqueue.p95_ms")
    trial_count = _integer(transfer.get("trial_count"), "trial_count")

    timing = _mapping(deadline.get("timing_evidence"), "timing_evidence")
    oracle = _mapping(deadline.get("one_layer_oracle"), "one_layer_oracle")
    windows = _mapping(
        deadline.get("observed_adjacent_layer_window_ms"),
        "observed_adjacent_layer_window_ms",
    )
    measurement_kind = deadline.get("measurement_kind")
    if not isinstance(measurement_kind, str):
        raise ValueError("deadline.measurement_kind must be a string")
    contention_measured = timing.get("contention_measured")
    live_runtime_measurement = timing.get("live_runtime_measurement")
    if not isinstance(contention_measured, bool) or not isinstance(
        live_runtime_measurement, bool
    ):
        raise ValueError("deadline evidence flags must be booleans")
    transfer_backend = timing.get("transfer_backend")
    window_backend = timing.get("window_backend")
    if not isinstance(transfer_backend, str) or not isinstance(
        window_backend, str
    ):
        raise ValueError("deadline backends must be strings")
    transfer_ms = _number(
        deadline.get("expert_transfer_ms"), "deadline.expert_transfer_ms"
    )
    no_prefetch_ms = _number(
        deadline.get("blocking_no_prefetch_ms_per_token"),
        "blocking_no_prefetch_ms_per_token",
    )
    oracle_ms = _number(
        oracle.get("residual_blocking_ms_per_token"),
        "residual_blocking_ms_per_token",
    )
    late_events = _integer(oracle.get("late_event_count"), "late_event_count")
    window_median = _number(windows.get("median"), "window.median")
    window_p95 = _number(windows.get("p95"), "window.p95")

    def policy_row(raw: object, *, prompt: bool) -> str:
        row = _mapping(raw, "breakdown row")
        row_static = _mapping(row.get("static_hotset"), "row.static_hotset")
        row_lru = _mapping(row.get("lru"), "row.lru")
        identifier = (
            row.get("conversation_id") if prompt else row.get("domain")
        )
        domain = row.get("domain")
        split = row.get("split", "all")
        if not all(isinstance(value, str) for value in (identifier, domain)):
            raise ValueError("breakdown identity must be a string")
        if not isinstance(split, str):
            raise ValueError("breakdown split must be a string")
        cells = f"<td>{escape(identifier)}</td>"
        if prompt:
            cells += f"<td>{escape(domain)}</td><td>{escape(split)}</td>"
        return (
            f"<tr>{cells}"
            f"<td>{_number(row_static.get('hit_rate'), 'row.static.hit_rate'):.2%}</td>"
            f"<td>{_number(row_lru.get('hit_rate'), 'row.lru.hit_rate'):.2%}</td>"
            f"<td>{_integer(row_static.get('miss_count'), 'row.static.miss_count'):,}</td>"
            f"<td>{_integer(row_lru.get('miss_count'), 'row.lru.miss_count'):,}</td></tr>"
        )

    domain_rows = "".join(
        policy_row(row, prompt=False)
        for row in _items(breakdown.get("per_domain"), "per_domain")
    )
    prompt_rows = "".join(
        policy_row(row, prompt=True)
        for row in _items(breakdown.get("per_prompt"), "per_prompt")
    )
    excluded = _items(breakdown.get("excluded_shards"), "excluded_shards")
    source_lines = "<br>".join(
        f"{escape(str(name))}: <code>{escape(str(path))}</code>"
        for name, path in sorted(sources.items(), key=lambda item: str(item[0]))
    )
    cache_mib = projected_cache_bytes / (1024 * 1024)
    encoded_range = (
        f"{encoded_min:,} bytes"
        if encoded_min == encoded_max
        else f"{encoded_min:,}-{encoded_max:,} bytes"
    )
    transfer_label = escape(
        measurement_kind.upper()
        .replace("_CROSS_", " CROSS-")
        .replace("_", " ")
    )
    return f"""
  <section><h2>Static-96 contract</h2><p><strong>{capacity} slots per target layer</strong>, not {capacity} slots globally. Layers {min(layers)}-{max(layers)} give {slot_count:,} slots. Each aligned slot is {slot_bytes:,} bytes, so the exact projected allocation is {projected_cache_bytes:,} bytes ({cache_mib:,.2f} MiB).</p><p>The inventory measured {object_count:,} layer-expert objects at {encoded_range}; {alignment}-byte packing produces {projected_slot_min:,}-byte slots. The arithmetic matches the packed layout exactly.</p><span class="tag estimated">PROJECTED ALLOCATION</span></section>
  <section><h2>Expanded held-out decode</h2><p>{_integer(aggregate.get('conversation_count'), 'conversation_count')} held-out conversations, {_integer(aggregate.get('token_count'), 'token_count'):,} decode tokens and {_integer(aggregate.get('expert_demand_count'), 'expert_demand_count'):,} expert demands. Static-96 hit {static_rate:.2%}; conversation-reset LRU hit {lru_rate:.2%}. Static-96 reduced cold bytes by {cold_reduction:.2%} ({static_misses:,} versus {lru_misses:,} misses), below the 20% practical-policy gate. {len(excluded)} failed shard was explicitly excluded.</p><h3>By domain</h3><div class="table-wrap"><table><thead><tr><th>Domain</th><th>Static-96</th><th>LRU</th><th>Static misses</th><th>LRU misses</th></tr></thead><tbody>{domain_rows}</tbody></table></div><h3>By prompt</h3><div class="table-wrap"><table><thead><tr><th>Prompt</th><th>Domain</th><th>Split</th><th>Static-96</th><th>LRU</th><th>Static misses</th><th>LRU misses</th></tr></thead><tbody>{prompt_rows}</tbody></table></div><span class="tag estimated">MEASURED ROUTING / ESTIMATED POLICY</span></section>
  <section><h2>Independent CUDA transfer</h2><p>{trial_count} idle-GPU trials. One {slot_bytes:,}-byte pinned H2D copy: p50 {pinned_p50:.4f} ms, p95 {pinned_p95:.4f} ms. Pageable H2D: p50 {pageable_p50:.4f} ms, p95 {pageable_p95:.4f} ms. Pinned sustained bandwidth was {sustained_gib_s:.2f} GiB/s; host API enqueue p50/p95 was {enqueue_p50:.4f}/{enqueue_p95:.4f} ms.</p><p class="muted">CUDA events measured copies on an idle default stream. This does not include model compute, contention, cache lookup, or replacement.</p><span class="tag">MEASURED CUDA MICROBENCHMARK</span></section>
  <section><h2>Deadline evidence boundary</h2><p><span class="tag estimated">{transfer_label}</span></p><p>The simulator combines {escape(transfer_backend)} transfer timing ({transfer_ms:.4f} ms) with {escape(window_backend)} windows (median {window_median:.4f} ms, p95 {window_p95:.4f} ms). It estimates {no_prefetch_ms:.4f} ms/token blocking without prefetch and {oracle_ms:.4f} ms/token residual under a perfect one-layer oracle, with {late_events:,} late events.</p><p class="warning"><strong>live_cache_enabled=false</strong>. CUDA contention measured: {str(contention_measured).lower()}. Live runtime measured: {str(live_runtime_measurement).lower()}. These simulator values are not runtime speedup, KV-cache, or CUDA deadline claims.</p></section>
  <section><h2>Physical evidence provenance</h2><p class="muted">{source_lines}</p></section>"""


def render_replay_report(
    recommendation: Mapping[str, object],
    replay: PolicyReplay,
    *,
    source_trace: Path | Sequence[Path],
    fit_trace: Path | Sequence[Path] | None = None,
    recommendation_source: Path,
    reproduction_command: str,
    max_events: int = 300,
    physical_evidence: Mapping[str, object] | None = None,
) -> str:
    """Render one portable report with no scripts or external assets."""

    if recommendation.get("schema_version") != "1.0.0":
        raise ValueError("recommendation schema_version must be '1.0.0'")
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    verdict = recommendation.get("verdict")
    if not isinstance(verdict, str):
        raise ValueError("recommendation verdict must be a string")
    hardware = _mapping(recommendation.get("hardware"), "hardware")
    replay_config = _mapping(recommendation.get("replay"), "replay")
    gpu_name = hardware.get("gpu_name")
    if not isinstance(gpu_name, str):
        raise ValueError("hardware.gpu_name must be a string")

    total_vram = _integer(hardware.get("total_vram_mib"), "total_vram_mib")
    peak_vram = _integer(
        hardware.get("measured_peak_vram_mib"), "measured_peak_vram_mib"
    )
    safety_reserve = _integer(
        hardware.get("safety_reserve_mib"), "safety_reserve_mib"
    )
    headroom = _integer(
        hardware.get("remaining_configurable_headroom_mib"),
        "remaining_configurable_headroom_mib",
    )
    selected_rate = _number(
        replay_config.get("estimated_hit_rate"), "estimated_hit_rate"
    )
    lru_rate = _number(
        replay_config.get("estimated_lru_hit_rate"),
        "estimated_lru_hit_rate",
    )
    reason_codes = recommendation.get("reason_codes")
    if not isinstance(reason_codes, list) or not all(
        isinstance(code, str) for code in reason_codes
    ):
        raise ValueError("reason_codes must be an array of strings")
    physical_sections = (
        ""
        if physical_evidence is None
        else _physical_evidence_sections(
            physical_evidence,
            live_cache_enabled=recommendation.get("live_cache_enabled"),
        )
    )

    cache_card = ""
    cache_detail = ""
    expert_cache_value = recommendation.get("expert_cache")
    if expert_cache_value is not None:
        expert_cache = _mapping(expert_cache_value, "expert_cache")
        projected_cache = _number(
            expert_cache.get("projected_cache_mib"),
            "expert_cache.projected_cache_mib",
        )
        remaining_after_cache = _number(
            expert_cache.get("remaining_headroom_after_cache_mib"),
            "expert_cache.remaining_headroom_after_cache_mib",
        )
        measured_transfer = _number(
            expert_cache.get("measured_expert_transfer_ms"),
            "expert_cache.measured_expert_transfer_ms",
        )
        estimated_sweep = _number(
            expert_cache.get(
                "estimated_serial_h2d_ms_per_layer_sweep"
            ),
            "expert_cache.estimated_serial_h2d_ms_per_layer_sweep",
        )
        fit_scope = expert_cache.get("fit_scope")
        if not isinstance(fit_scope, str) or not fit_scope:
            raise ValueError("expert_cache.fit_scope must be a string")
        cache_card = (
            '<div class="card"><div class="label">Projected expert cache</div>'
            f'<div class="metric">{projected_cache:,.2f} MiB</div>'
            f'<span class="tag estimated">{escape(fit_scope.upper().replace("_", "-"))}</span></div>'
        )
        cache_detail = (
            "<section><h2>Projected cache envelope</h2>"
            f"<p>Replay allocation {projected_cache:,.2f} MiB &middot; "
            f"remaining configurable headroom {remaining_after_cache:,.2f} MiB. "
            f"Measured pinned expert transfer {measured_transfer:.4f} ms; "
            f"estimated serialized H2D per selected-layer sweep "
            f"{estimated_sweep:.4f} ms.</p>"
            '<p class="muted">The allocation and sweep are estimates; live caching remains disabled.</p></section>'
        )

    rows: list[str] = []
    for event in replay.timeline[:max_events]:
        ready = ", ".join(str(value) for value in event.ready_expert_ids) or "none"
        blocking = (
            ", ".join(str(value) for value in event.blocking_expert_ids)
            or "none"
        )
        rows.append(
            "<tr>"
            f'<td><span class="status {escape(event.status)}">{escape(event.status.upper())}</span></td>'
            f"<td>{event.token_index}</td><td>{event.layer_id}</td>"
            f"<td>{escape(event.phase)}</td><td>{escape(ready)}</td>"
            f"<td>{escape(blocking)}</td></tr>"
        )
    omitted = len(replay.timeline) - len(rows)
    reason_items = "".join(
        f"<li><code>{escape(str(code))}</code></li>" for code in reason_codes
    )
    timeline_note = (
        f"Showing {len(rows):,} of {len(replay.timeline):,} events"
        + (f"; {omitted:,} omitted from this bounded view." if omitted else ".")
    )
    source_paths = (
        (source_trace,)
        if isinstance(source_trace, Path)
        else tuple(source_trace)
    )
    if not source_paths:
        raise ValueError("source_trace must contain at least one path")
    trace_lines = "<br>".join(
        f"<code>{escape(str(path.resolve()))}</code>"
        for path in source_paths
    )
    fit_lines = ""
    if fit_trace is not None:
        fit_paths = (
            (fit_trace,) if isinstance(fit_trace, Path) else tuple(fit_trace)
        )
        if not fit_paths:
            raise ValueError("fit_trace must contain at least one path")
        rendered_fit_paths = "<br>".join(
            f"<code>{escape(str(path.resolve()))}</code>"
            for path in fit_paths
        )
        fit_lines = f"<br>Training traces:<br>{rendered_fit_paths}"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ExpertFlow Observatory replay</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0a0f18; --panel:#121a27; --line:#26354b; --text:#e8eef8; --muted:#98a8bd; --cyan:#51d5ff; --amber:#ffbe55; --green:#61e6a5; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 15% 0%,#162944 0,var(--bg) 42%); color:var(--text); font:15px/1.5 ui-sans-serif,Segoe UI,sans-serif; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:36px auto 72px; }}
    h1 {{ margin:0; font-size:clamp(30px,5vw,54px); letter-spacing:-.04em; }}
    h2 {{ margin:0 0 16px; font-size:20px; }}
    .eyebrow,.label {{ color:var(--cyan); font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }}
    .lede {{ color:var(--muted); max-width:760px; font-size:17px; }}
    .verdict {{ display:inline-block; margin:14px 0; padding:7px 11px; border:1px solid var(--amber); color:var(--amber); border-radius:999px; font-weight:800; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:24px 0; }}
    .card,section {{ background:color-mix(in srgb,var(--panel) 94%,transparent); border:1px solid var(--line); border-radius:14px; box-shadow:0 18px 50px #0005; }}
    .card {{ padding:18px; }} .metric {{ font-size:28px; font-weight:800; margin-top:5px; }}
    section {{ padding:22px; margin-top:16px; overflow:hidden; }}
    .tag {{ padding:3px 7px; border-radius:6px; background:#1b3048; color:var(--cyan); font-size:11px; font-weight:800; }}
    .tag.estimated {{ color:var(--amber); }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }} th,td {{ padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .table-wrap {{ overflow:auto; }} .status {{ font-size:11px; font-weight:800; }} .status.ready {{ color:var(--green); }} .status.blocking {{ color:var(--amber); }}
    code {{ color:#c4dcff; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#090d14; border:1px solid var(--line); padding:14px; border-radius:9px; }}
    .muted {{ color:var(--muted); }} .warning {{ color:var(--amber); }} h3 {{ margin:24px 0 10px; font-size:15px; }}
  </style>
</head>
<body><main>
  <div class="eyebrow">ExpertFlow Local / causal replay</div>
  <h1>Sparse routing, made inspectable.</h1>
  <div class="verdict">{escape(verdict)}</div>
  <p class="lede">The router trace is measured. Cache outcomes are replay estimates. Live caching remains disabled until the listed evidence gaps are closed.</p>
  <div class="grid">
    <div class="card"><div class="label">GPU</div><div class="metric">{escape(gpu_name)}</div><span class="tag">MEASURED</span></div>
    <div class="card"><div class="label">Peak VRAM</div><div class="metric">{peak_vram:,} MiB</div><span class="tag">MEASURED</span></div>
    <div class="card"><div class="label">Configurable headroom</div><div class="metric">{headroom:,} MiB</div><span class="tag">MEASURED</span></div>
    <div class="card"><div class="label">Replay hit rate</div><div class="metric">{selected_rate:.2%}</div><span class="tag estimated">ESTIMATED</span></div>
    {cache_card}
  </div>
  <section><h2>Memory envelope</h2><p>Total {total_vram:,} MiB &middot; measured peak {peak_vram:,} MiB &middot; safety reserve {safety_reserve:,} MiB &middot; remaining {headroom:,} MiB.</p></section>
  {cache_detail}
  {physical_sections}
  <section><h2>Policy decision</h2><p><strong>{escape(replay.policy)}</strong> at {replay.capacity_per_layer} slots/layer: {selected_rate:.2%} estimated selection hits. LRU comparison: {lru_rate:.2%}.</p><p>{replay.hit_count:,} ready selections; {replay.miss_count:,} blocking selections across {replay.event_count:,} token/layer events.</p><ul>{reason_items}</ul></section>
  <section><h2>Causal timeline</h2><p class="muted">{escape(timeline_note)}</p><div class="table-wrap"><table><thead><tr><th>Outcome</th><th>Token</th><th>Layer</th><th>Phase</th><th>Ready experts</th><th>Blocking experts</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
  <section><h2>Provenance and reproduction</h2><p class="muted">Trace schema: <code>1.0.0</code><br>Recommendation schema: <code>1.0.0</code><br>Evaluation traces:<br>{trace_lines}{fit_lines}<br>Recommendation: <code>{escape(str(recommendation_source.resolve()))}</code></p><pre><code>{escape(reproduction_command)}</code></pre></section>
</main></body></html>"""
