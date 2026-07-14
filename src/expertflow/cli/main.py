"""Public ExpertFlow command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path

from expertflow.analysis.cache_sim import simulate_policies
from expertflow.analysis.capacity_curve import (
    build_capacity_curve,
    build_held_out_capacity_curve,
)
from expertflow.analysis.deadline import evaluate_one_layer_oracle
from expertflow.analysis.profile import summarize_routing
from expertflow.analysis.replay import replay_policy
from expertflow.doctor import collect_doctor_report
from expertflow.recommendation import build_recommendation
from expertflow.reporting import render_replay_report
from expertflow.runtime.baseline import BaselineRunConfig
from expertflow.runtime.cuda_transfer import benchmark_cuda_transfers
from expertflow.runtime.measurement import run_measured_baseline
from expertflow.trace.io import load_router_events
from expertflow.trace.parity import compare_token_sequences


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

    parity = commands.add_parser(
        "parity",
        help="Compare deterministic token sequences with and without tracing.",
    )
    parity.add_argument("baseline", type=Path)
    parity.add_argument("instrumented", type=Path)
    parity.add_argument("--output", type=Path, required=True)

    recommend = commands.add_parser(
        "recommend",
        help="Create an evidence-bounded machine-specific recommendation.",
    )
    recommend.add_argument("--doctor", type=Path, required=True)
    recommend.add_argument("--baseline", type=Path, required=True)
    recommend.add_argument("--profile", type=Path, required=True)
    recommend.add_argument("--simulation", type=Path, required=True)
    recommend.add_argument("--capacity-curve", type=Path)
    recommend.add_argument("--output", type=Path, required=True)
    recommend.add_argument("--safety-reserve-mib", type=int, default=1024)

    replay = commands.add_parser(
        "replay", help="Render a standalone causal policy replay report."
    )
    replay.add_argument("trace", type=Path, nargs="+")
    replay.add_argument("--recommendation", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--max-events", type=int, default=300)
    replay.add_argument("--phase", choices=("prefill", "decode"))
    replay.add_argument("--max-layer", type=int)
    replay.add_argument("--fit-trace", type=Path, action="append")
    replay.add_argument("--fit-phase", choices=("prefill", "decode"))

    simulate = commands.add_parser(
        "simulate", help="Compare estimated cache policies over a router trace."
    )
    simulate.add_argument("trace", type=Path)
    simulate.add_argument("--capacity-per-layer", type=int, required=True)
    simulate.add_argument("--output", type=Path, required=True)

    transfer = commands.add_parser(
        "transfer-benchmark",
        help="Measure CUDA host-to-device transfer latency.",
    )
    transfer.add_argument("--cudart", type=Path, required=True)
    transfer.add_argument(
        "--payload-bytes", type=int, action="append", required=True
    )
    transfer.add_argument("--batches", type=int, default=30)
    transfer.add_argument("--copies-per-batch", type=int, default=50)
    transfer.add_argument("--warmup-copies", type=int, default=10)
    transfer.add_argument("--device", type=int, default=0)
    transfer.add_argument("--output", type=Path, required=True)

    curve = commands.add_parser(
        "capacity-curve",
        help="Estimate cache policy curves over one or more router traces.",
    )
    curve.add_argument("trace", type=Path, nargs="+")
    curve.add_argument("--phase", choices=("prefill", "decode"))
    curve.add_argument("--max-layer", type=int)
    curve.add_argument(
        "--capacity", type=int, action="append", required=True
    )
    curve.add_argument("--slot-bytes", type=int, required=True)
    curve.add_argument("--expert-transfer-ms", type=float, required=True)
    curve.add_argument("--output", type=Path, required=True)

    heldout = commands.add_parser(
        "heldout-curve",
        help="Fit static residents on training traces and score held-out traces.",
    )
    heldout.add_argument(
        "--train", type=Path, action="append", required=True
    )
    heldout.add_argument(
        "--eval", type=Path, action="append", required=True
    )
    heldout.add_argument("--phase", choices=("prefill", "decode"))
    heldout.add_argument("--train-phase", choices=("prefill", "decode"))
    heldout.add_argument("--eval-phase", choices=("prefill", "decode"))
    heldout.add_argument("--max-layer", type=int)
    heldout.add_argument(
        "--capacity", type=int, action="append", required=True
    )
    heldout.add_argument("--slot-bytes", type=int, required=True)
    heldout.add_argument("--expert-transfer-ms", type=float, required=True)
    heldout.add_argument("--output", type=Path, required=True)

    deadline = commands.add_parser(
        "deadline-eval",
        help="Evaluate a backend-specific oracle one-layer transfer bound.",
    )
    deadline.add_argument(
        "--train", type=Path, action="append", required=True
    )
    deadline.add_argument(
        "--eval", type=Path, action="append", required=True
    )
    deadline.add_argument(
        "--train-phase", choices=("prefill", "decode"), required=True
    )
    deadline.add_argument(
        "--eval-phase", choices=("prefill", "decode"), required=True
    )
    deadline.add_argument("--max-layer", type=int)
    deadline.add_argument("--capacity", type=int, required=True)
    deadline.add_argument("--expert-transfer-ms", type=float, required=True)
    deadline.add_argument("--output", type=Path, required=True)
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


def _run_parity(args: argparse.Namespace) -> int:
    report = compare_token_sequences(args.baseline, args.instrumented)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return int(
        not (report["prompt_matches"] and report["generated_matches"])
    )


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return value


def _run_recommend(args: argparse.Namespace) -> int:
    source_paths = {
        "doctor": args.doctor.resolve(),
        "baseline": args.baseline.resolve(),
        "profile": args.profile.resolve(),
        "simulation": args.simulation.resolve(),
    }
    if args.capacity_curve is not None:
        source_paths["capacity_curve"] = args.capacity_curve.resolve()
    report = build_recommendation(
        _load_json_object(source_paths["doctor"], "doctor"),
        _load_json_object(source_paths["baseline"], "baseline"),
        _load_json_object(source_paths["profile"], "profile"),
        _load_json_object(source_paths["simulation"], "simulation"),
        capacity_curve=(
            _load_json_object(
                source_paths["capacity_curve"], "capacity_curve"
            )
            if "capacity_curve" in source_paths
            else None
        ),
        safety_reserve_mib=args.safety_reserve_mib,
    )
    report["sources"] = {
        name: str(path) for name, path in source_paths.items()
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    if args.max_layer is not None and args.max_layer < 0:
        raise ValueError("max_layer must be non-negative")
    sources = [source.resolve() for source in args.trace]
    training_sources = [
        source.resolve() for source in (args.fit_trace or [])
    ]
    if args.fit_phase is not None and not training_sources:
        raise ValueError("fit_phase requires at least one fit_trace")
    training_phase = args.fit_phase or args.phase
    recommendation_path = args.recommendation.resolve()
    recommendation = _load_json_object(
        recommendation_path, "recommendation"
    )
    replay_config = recommendation.get("replay")
    if not isinstance(replay_config, dict):
        raise ValueError("recommendation replay must be an object")
    policy = replay_config.get("policy")
    if policy not in {"reactive", "static_hotset", "lru"}:
        raise ValueError("recommendation replay policy is unsupported")
    capacity = replay_config.get("capacity_per_layer")
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise ValueError("recommendation capacity_per_layer must be an integer")

    events = [
        event
        for source in sources
        for event in load_router_events(source)
        if (args.phase is None or event.phase == args.phase)
        and (args.max_layer is None or event.layer_id <= args.max_layer)
    ]
    if not events:
        raise ValueError("replay selection produced no events")
    training_events = [
        event
        for source in training_sources
        for event in load_router_events(source)
        if (training_phase is None or event.phase == training_phase)
        and (args.max_layer is None or event.layer_id <= args.max_layer)
    ]
    replay = replay_policy(
        events,
        policy=policy,
        capacity_per_layer=capacity,
        static_training_events=(
            training_events if training_sources else None
        ),
    )
    output = args.output.resolve()
    trace_arguments = " ".join(f'"{source}"' for source in sources)
    selection_arguments = ""
    if args.phase is not None:
        selection_arguments += f" --phase {args.phase}"
    if args.max_layer is not None:
        selection_arguments += f" --max-layer {args.max_layer}"
    fit_arguments = "".join(
        f' --fit-trace "{source}"' for source in training_sources
    )
    if args.fit_phase is not None:
        fit_arguments += f" --fit-phase {args.fit_phase}"
    command = (
        f"expertflow replay {trace_arguments}{selection_arguments}"
        f"{fit_arguments} "
        f'--recommendation "{recommendation_path}" --output "{output}"'
    )
    rendered = render_replay_report(
        recommendation,
        replay,
        source_trace=sources,
        fit_trace=training_sources or None,
        recommendation_source=recommendation_path,
        reproduction_command=command,
        max_events=args.max_events,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
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


def _run_transfer_benchmark(args: argparse.Namespace) -> int:
    report = benchmark_cuda_transfers(
        args.cudart.resolve(),
        payload_bytes=tuple(args.payload_bytes),
        batches=args.batches,
        copies_per_batch=args.copies_per_batch,
        warmup_copies=args.warmup_copies,
        device=args.device,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def _run_capacity_curve(args: argparse.Namespace) -> int:
    if args.max_layer is not None and args.max_layer < 0:
        raise ValueError("max_layer must be non-negative")
    sources = [path.resolve() for path in args.trace]
    events = [
        event
        for source in sources
        for event in load_router_events(source)
        if (args.phase is None or event.phase == args.phase)
        and (args.max_layer is None or event.layer_id <= args.max_layer)
    ]
    report = build_capacity_curve(
        events,
        capacities=tuple(args.capacity),
        slot_bytes=args.slot_bytes,
        expert_transfer_ms=args.expert_transfer_ms,
    )
    report["source_traces"] = [str(source) for source in sources]
    report["selection"] = {
        "phase": args.phase,
        "max_layer": args.max_layer,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def _run_heldout_curve(args: argparse.Namespace) -> int:
    if args.max_layer is not None and args.max_layer < 0:
        raise ValueError("max_layer must be non-negative")
    training_sources = [path.resolve() for path in args.train]
    evaluation_sources = [path.resolve() for path in args.eval]
    if args.phase is not None and (
        args.train_phase is not None or args.eval_phase is not None
    ):
        raise ValueError(
            "phase cannot be combined with train_phase or eval_phase"
        )
    training_phase = args.train_phase or args.phase
    evaluation_phase = args.eval_phase or args.phase

    def selected(sources: list[Path], phase: str | None):
        return [
            event
            for source in sources
            for event in load_router_events(source)
            if (phase is None or event.phase == phase)
            and (args.max_layer is None or event.layer_id <= args.max_layer)
        ]

    report = build_held_out_capacity_curve(
        selected(training_sources, training_phase),
        selected(evaluation_sources, evaluation_phase),
        capacities=tuple(args.capacity),
        slot_bytes=args.slot_bytes,
        expert_transfer_ms=args.expert_transfer_ms,
    )
    report["training_source_traces"] = [
        str(source) for source in training_sources
    ]
    report["evaluation_source_traces"] = [
        str(source) for source in evaluation_sources
    ]
    report["selection"] = {
        "training_phase": training_phase,
        "evaluation_phase": evaluation_phase,
        "max_layer": args.max_layer,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def _run_deadline_eval(args: argparse.Namespace) -> int:
    if args.max_layer is not None and args.max_layer < 0:
        raise ValueError("max_layer must be non-negative")
    training_sources = [path.resolve() for path in args.train]
    evaluation_sources = [path.resolve() for path in args.eval]

    def selected(source: Path, phase: str):
        return [
            event
            for event in load_router_events(source)
            if event.phase == phase
            and (args.max_layer is None or event.layer_id <= args.max_layer)
        ]

    training_events = [
        event
        for source in training_sources
        for event in selected(source, args.train_phase)
    ]
    evaluation_traces = [
        selected(source, args.eval_phase) for source in evaluation_sources
    ]
    report = evaluate_one_layer_oracle(
        training_events,
        evaluation_traces,
        capacity_per_layer=args.capacity,
        expert_transfer_ms=args.expert_transfer_ms,
    )
    report["training_source_traces"] = [
        str(source) for source in training_sources
    ]
    report["evaluation_source_traces"] = [
        str(source) for source in evaluation_sources
    ]
    report["selection"] = {
        "training_phase": args.train_phase,
        "evaluation_phase": args.eval_phase,
        "max_layer": args.max_layer,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    if args.command == "parity":
        return _run_parity(args)
    if args.command == "recommend":
        return _run_recommend(args)
    if args.command == "replay":
        return _run_replay(args)
    if args.command == "simulate":
        return _run_simulate(args)
    if args.command == "transfer-benchmark":
        return _run_transfer_benchmark(args)
    if args.command == "capacity-curve":
        return _run_capacity_curve(args)
    if args.command == "heldout-curve":
        return _run_heldout_curve(args)
    if args.command == "deadline-eval":
        return _run_deadline_eval(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
