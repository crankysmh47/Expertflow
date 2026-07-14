"""Public ExpertFlow command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path

from expertflow.analysis.profile import summarize_routing
from expertflow.doctor import collect_doctor_report
from expertflow.trace.io import load_router_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expertflow",
        description="Profile and explain sparse-MoE routing on local hardware.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

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
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _run_doctor(args)
    if args.command == "profile":
        return _run_profile(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
