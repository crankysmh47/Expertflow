"""Public ExpertFlow command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path

from expertflow.analysis.cache_sim import simulate_policies
from expertflow.analysis.profile import summarize_routing
from expertflow.doctor import collect_doctor_report
from expertflow.runtime.baseline import BaselineRunConfig
from expertflow.runtime.measurement import run_measured_baseline
from expertflow.trace.io import load_router_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expertflow",
        description="Profile and explain sparse-MoE routing on local hardware.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser(
        "baseline", help="Run and measure an unmodified llama.cpp baseline."
    )
    baseline.add_argument("--runtime", type=Path, required=True)
    baseline.add_argument("--model", type=Path, required=True)
    baseline.add_argument("--model-sha256", required=True)
    baseline.add_argument("--prompt-file", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--gpu-layers", default="auto")
    baseline.add_argument("--ctx-size", type=int, default=1024)
    baseline.add_argument("--predict", type=int, default=32)
    baseline.add_argument("--threads", type=int, default=12)

    doctor = commands.add_parser(
        "doctor", help="Record hardware, storage, and toolchain readiness."
    )
    doctor.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(r"C:\models\expertflow"),
    )
    doctor.add_argument("--output", type=Path)

    profile = commands.add_parser(
        "profile", help="Create a measured locality profile from a router trace."
    )
    profile.add_argument("trace", type=Path)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument(
        "--static-budget",
        type=int,
        action="append",
        help="Resident experts per layer; repeat to produce a hit curve.",
    )

    simulate = commands.add_parser(
        "simulate", help="Compare estimated cache policies over a router trace."
    )
    simulate.add_argument("trace", type=Path)
    simulate.add_argument("--capacity-per-layer", type=int, required=True)
    simulate.add_argument("--output", type=Path, required=True)
    return parser


def _run_baseline(args: argparse.Namespace) -> int:
    digest = args.model_sha256.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("model_sha256 must be 64 hexadecimal characters")

    output_dir = args.output_dir.resolve()
    config = BaselineRunConfig(
        executable=args.runtime.resolve(),
        model=args.model.resolve(),
        prompt=args.prompt_file.read_text(encoding="utf-8").strip(),
        log_file=output_dir / "llama.log",
        gpu_layers=args.gpu_layers,
        context_size=args.ctx_size,
        predict_tokens=args.predict,
        threads=args.threads,
    )
    manifest_path = output_dir / "manifest.json"
    result = run_measured_baseline(
        config,
        model_sha256=digest,
        manifest_path=manifest_path,
    )
    print(manifest_path)
    return int(result["return_code"])


def _run_doctor(args: argparse.Namespace) -> int:
    report = collect_doctor_report(args.artifact_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return 0

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)
    return 0


def _run_profile(args: argparse.Namespace) -> int:
    budgets = tuple(args.static_budget or (1, 2, 4, 8))
    source = args.trace.resolve()
    profile = summarize_routing(
        load_router_events(source), static_budgets=budgets
    )
    report = {
        "schema_version": "1.0.0",
        "measurement_kind": "measured",
        "source_trace": str(source),
        "profile": asdict(profile),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


def _run_simulate(args: argparse.Namespace) -> int:
    source = args.trace.resolve()
    simulation = simulate_policies(
        list(load_router_events(source)),
        capacity_per_layer=args.capacity_per_layer,
    )
    report = {
        "schema_version": "1.0.0",
        "measurement_kind": simulation.measurement_kind,
        "source_trace": str(source),
        "simulation": asdict(simulation),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        return _run_baseline(args)
    if args.command == "doctor":
        return _run_doctor(args)
    if args.command == "profile":
        return _run_profile(args)
    if args.command == "simulate":
        return _run_simulate(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
