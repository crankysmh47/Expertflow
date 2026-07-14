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


def render_replay_report(
    recommendation: Mapping[str, object],
    replay: PolicyReplay,
    *,
    source_trace: Path | Sequence[Path],
    fit_trace: Path | Sequence[Path] | None = None,
    recommendation_source: Path,
    reproduction_command: str,
    max_events: int = 300,
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
            f"<p>Replay allocation {projected_cache:,.2f} MiB · "
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
    .muted {{ color:var(--muted); }}
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
  <section><h2>Memory envelope</h2><p>Total {total_vram:,} MiB · measured peak {peak_vram:,} MiB · safety reserve {safety_reserve:,} MiB · remaining {headroom:,} MiB.</p></section>
  {cache_detail}
  <section><h2>Policy decision</h2><p><strong>{escape(replay.policy)}</strong> at {replay.capacity_per_layer} slots/layer: {selected_rate:.2%} estimated selection hits. LRU comparison: {lru_rate:.2%}.</p><p>{replay.hit_count:,} ready selections; {replay.miss_count:,} blocking selections across {replay.event_count:,} token/layer events.</p><ul>{reason_items}</ul></section>
  <section><h2>Causal timeline</h2><p class="muted">{escape(timeline_note)}</p><div class="table-wrap"><table><thead><tr><th>Outcome</th><th>Token</th><th>Layer</th><th>Phase</th><th>Ready experts</th><th>Blocking experts</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
  <section><h2>Provenance and reproduction</h2><p class="muted">Trace schema: <code>1.0.0</code><br>Recommendation schema: <code>1.0.0</code><br>Evaluation traces:<br>{trace_lines}{fit_lines}<br>Recommendation: <code>{escape(str(recommendation_source.resolve()))}</code></p><pre><code>{escape(reproduction_command)}</code></pre></section>
</main></body></html>"""
