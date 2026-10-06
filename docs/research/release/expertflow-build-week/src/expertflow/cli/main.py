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
from expertflow.analysis.heldout_breakdown import (
    build_heldout_breakdown,
    load_collection_breakdown_inputs,
)
from expertflow.analysis.profile import summarize_routing
from expertflow.analysis.replay import replay_policy
from expertflow.collection import CollectionConfig, collect_trace_pairs
from expertflow.doctor import collect_doctor_report
from expertflow.recommendation import build_recommendation
from expertflow.reporting import render_replay_report
from expertflow.runtime.baseline import BaselineRunConfig
from expertflow.runtime.cuda_transfer import (
    aggregate_cuda_transfer_trials,
    benchmark_cuda_transfers,
)
from expertflow.runtime.measurement import run_measured_baseline
from expertflow.trace.io import load_router_events
from expertflow.trace.parity import compare_token_sequences
from expertflow.product.commands import (
    DEFAULT_DEPLOYMENT,
    build_runtime_command,
    compare_report,
    doctor_report,
    load_json,
    optimize_deployment,
    profile_report,
    replay_report,
    sha256_file,
)
import os
import subprocess


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

    collect = commands.add_parser(
        "collect-pairs",
        help="Collect resumable tracing-disabled/tracing-enabled prompt pairs.",
    )
    collect.add_argument("--corpus", type=Path, required=True)
    collect.add_argument("--probe", type=Path, required=True)
    collect.add_argument("--model", type=Path, required=True)
    collect.add_argument("--model-sha256", required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--predict", type=int, default=64)
    collect.add_argument("--gpu-layers", type=int, default=10)
    collect.add_argument("--threads", type=int, default=12)

    doctor = commands.add_parser(
        "doctor", help="Record hardware, storage, and toolchain readiness."
    )
    doctor.add_argument(
        "--artifact-root",
        type=Path,
        default=Path.cwd(),
    )
    doctor.add_argument("--output", type=Path)
    doctor.add_argument("--model", type=Path)
    doctor.add_argument("--runtime", type=Path)
    doctor.add_argument("--server", type=Path)

    profile = commands.add_parser(
        "profile", help="Create a measured locality profile from a router trace."
    )
    profile.add_argument("trace", type=Path)
    profile.add_argument("--output", type=Path)
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
    replay.add_argument("--heldout-breakdown", type=Path)
    replay.add_argument("--expert-layout", type=Path)
    replay.add_argument("--transfer-evidence", type=Path)
    replay.add_argument("--deadline-evidence", type=Path)

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
    transfer.add_argument("--single-copy-samples", type=int, default=200)
    transfer.add_argument("--device", type=int, default=0)
    transfer.add_argument("--output", type=Path, required=True)

    transfer_aggregate = commands.add_parser(
        "transfer-aggregate",
        help="Pool raw CUDA transfer samples from independent trials.",
    )
    transfer_aggregate.add_argument("trial", type=Path, nargs="+")
    transfer_aggregate.add_argument("--output", type=Path, required=True)

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

    breakdown = commands.add_parser(
        "heldout-breakdown",
        help="Report held-out policy results by conversation and domain.",
    )
    breakdown.add_argument(
        "--collection-manifest", type=Path, required=True
    )
    breakdown.add_argument(
        "--phase", choices=("prefill", "decode"), required=True
    )
    breakdown.add_argument(
        "--eval-phase",
        choices=("prefill", "decode"),
        help="Evaluation phase; defaults to the training --phase.",
    )
    breakdown.add_argument("--max-layer", type=int)
    breakdown.add_argument(
        "--exclude-failed-shards",
        action="store_true",
        help="Exclude failed parity shards and list every exclusion in output.",
    )
    breakdown.add_argument("--capacity", type=int, required=True)
    breakdown.add_argument("--slot-bytes", type=int, required=True)
    breakdown.add_argument(
        "--expert-transfer-ms", type=float, required=True
    )
    breakdown.add_argument("--output", type=Path, required=True)

    deadline = commands.add_parser(
        "deadline-eval",
        help="Evaluate a backend-specific oracle one-layer transfer bound.",
    )
    deadline.add_argument(
        "--train", type=Path, action="append"
    )
    deadline.add_argument(
        "--eval", type=Path, action="append"
    )
    deadline.add_argument("--collection-manifest", type=Path)
    deadline.add_argument("--exclude-failed-shards", action="store_true")
    deadline.add_argument(
        "--train-phase", choices=("prefill", "decode"), required=True
    )
    deadline.add_argument(
        "--eval-phase", choices=("prefill", "decode"), required=True
    )
    deadline.add_argument("--max-layer", type=int)
    deadline.add_argument("--capacity", type=int, required=True)
    deadline.add_argument("--expert-transfer-ms", type=float, required=True)
    deadline.add_argument("--transfer-backend")
    deadline.add_argument("--window-backend")
    deadline.add_argument("--transfer-statistic")
    deadline.add_argument("--transfer-source", type=Path)
    deadline.add_argument("--window-source", type=Path)
    deadline.add_argument("--output", type=Path, required=True)

    optimize = commands.add_parser("optimize", help="Create a measured deployment manifest.")
    optimize.add_argument("model", type=Path)
    optimize.add_argument("--goal", choices=("latency", "throughput", "context", "agentic"), required=True)
    optimize.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="Launch inference from a deployment manifest.")
    run.add_argument("deployment", type=Path)
    run.add_argument("--runtime", type=Path)
    run.add_argument("--model", type=Path)
    run.add_argument("--dry-run", action="store_true")

    serve = commands.add_parser("serve", help="Launch an OpenAI-compatible llama-server deployment.")
    serve.add_argument("deployment", type=Path)
    serve.add_argument("--server", type=Path)
    serve.add_argument("--model", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--dry-run", action="store_true")

    compare = commands.add_parser("compare", help="Compare stock and ExpertFlow recorded evidence.")
    compare.add_argument("deployment", type=Path, nargs="?", default=DEFAULT_DEPLOYMENT)

    demo = commands.add_parser("demo", help="Replay the committed product evidence.")
    demo.add_argument("--replay", action="store_true", required=True)
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
    model = _resolve_product_path(args.model, "EXPERTFLOW_MODEL_PATH")
    runtime = _resolve_product_path(args.runtime, "EXPERTFLOW_LLAMA_CLI")
    server = _resolve_product_path(args.server, "EXPERTFLOW_LLAMA_SERVER")
    report = doctor_report(model, runtime, server)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return int(report["exit_code"])

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)
    return int(report["exit_code"])


def _run_collect_pairs(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    report = collect_trace_pairs(
        args.corpus.resolve(),
        CollectionConfig(
            probe=args.probe.resolve(),
            model=args.model.resolve(),
            model_sha256=args.model_sha256,
            output_dir=output_dir,
            n_predict=args.predict,
            gpu_layers=args.gpu_layers,
            threads=args.threads,
        ),
    )
    print(output_dir / "collection-manifest.json")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("collection report summary must be an object")
    failed = summary.get("failed")
    if isinstance(failed, bool) or not isinstance(failed, int):
        raise ValueError("collection failed count must be an integer")
    return int(failed > 0)


def _run_profile(args: argparse.Namespace) -> int:
    if args.output is None or args.trace.suffix.lower() == ".gguf":
        print(json.dumps(profile_report(args.trace), indent=2, sort_keys=True))
        return 0
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


def _resolve_product_path(explicit: Path | None, env_name: str, fallback: str | None = None) -> Path | None:
    if explicit is not None:
        return explicit
    value = os.environ.get(env_name) or fallback
    return Path(value) if value else None


def _run_optimize(args: argparse.Namespace) -> int:
    code, report, deployment = optimize_deployment(args.model, args.goal)
    if deployment is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["output"] = str(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


def _run_product_inference(args: argparse.Namespace) -> int:
    deployment = load_json(args.deployment)
    runtime = _resolve_product_path(args.runtime, "EXPERTFLOW_LLAMA_CLI")
    model = _resolve_product_path(args.model, "EXPERTFLOW_MODEL_PATH", deployment.get("model", {}).get("path"))
    if runtime is None or model is None:
        print(json.dumps({"status": "failure", "reason": "runtime and model paths are required"}, indent=2))
        return 2
    command, environment = build_runtime_command(deployment, runtime=runtime, model=model)
    report = {"status": "pass" if args.dry_run else "running", "command": command, "selected_expert_layers": deployment["placement"]["static_expert_layers"]}
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0
    return subprocess.run(command, env=environment, check=False).returncode


def _run_product_server(args: argparse.Namespace) -> int:
    deployment = load_json(args.deployment)
    server = _resolve_product_path(args.server, "EXPERTFLOW_LLAMA_SERVER")
    model = _resolve_product_path(args.model, "EXPERTFLOW_MODEL_PATH", deployment.get("model", {}).get("path"))
    if server is None or model is None:
        print(json.dumps({"status": "failure", "reason": "server and model paths are required"}, indent=2))
        return 2
    base = f"http://{args.host}:{args.port}"
    command = [str(server), "-m", str(model), "--host", args.host, "--port", str(args.port), "-c", str(deployment.get("context", 2048)), "-np", str(deployment.get("parallel_slots", 1)), "-ngl", "99", "--cpu-moe"]
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in deployment["environment"].items()})
    print(json.dumps({"status": "pass" if args.dry_run else "running", "base_url": base + "/v1", "health_url": base + "/health", "model": deployment["id"], "context": deployment.get("context", 2048), "parallel_slots": deployment.get("parallel_slots", 1), "selected_expert_layers": deployment["placement"]["static_expert_layers"], "expected_peak_vram_mib": deployment["expected_peak_vram_mib"], "command": command}, indent=2))
    if args.dry_run:
        return 0
    return subprocess.run(command, env=environment, check=False).returncode


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
    physical_arguments = {
        "heldout_breakdown": args.heldout_breakdown,
        "expert_layout": args.expert_layout,
        "transfer": args.transfer_evidence,
        "deadline": args.deadline_evidence,
    }
    supplied_physical = {
        name: path.resolve()
        for name, path in physical_arguments.items()
        if path is not None
    }
    if supplied_physical and len(supplied_physical) != len(physical_arguments):
        raise ValueError(
            "physical replay evidence requires heldout breakdown, expert "
            "layout, transfer evidence, and deadline evidence"
        )
    physical_evidence = None
    if supplied_physical:
        physical_evidence = {
            name: _load_json_object(path, name)
            for name, path in supplied_physical.items()
        }
        physical_evidence["sources"] = {
            name: str(path) for name, path in supplied_physical.items()
        }
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
    physical_cli_names = {
        "heldout_breakdown": "heldout-breakdown",
        "expert_layout": "expert-layout",
        "transfer": "transfer-evidence",
        "deadline": "deadline-evidence",
    }
    physical_cli_arguments = "".join(
        f' --{physical_cli_names[name]} "{path}"'
        for name, path in supplied_physical.items()
    )
    command = (
        f"expertflow replay {trace_arguments}{selection_arguments}"
        f"{fit_arguments}{physical_cli_arguments} "
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
        physical_evidence=physical_evidence,
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
        single_copy_samples=args.single_copy_samples,
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


def _run_transfer_aggregate(args: argparse.Namespace) -> int:
    sources = tuple(path.resolve() for path in args.trial)
    report = aggregate_cuda_transfer_trials(
        [_load_json_object(path, "transfer trial") for path in sources],
        source_paths=tuple(str(path) for path in sources),
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

    def selected_source(source: Path, phase: str | None):
        return [
            event
            for event in load_router_events(source)
            if (phase is None or event.phase == phase)
            and (args.max_layer is None or event.layer_id <= args.max_layer)
        ]

    training_events = [
        event
        for source in training_sources
        for event in selected_source(source, training_phase)
    ]
    evaluation_groups = tuple(
        tuple(selected_source(source, evaluation_phase))
        for source in evaluation_sources
    )
    evaluation_events = [
        event for group in evaluation_groups for event in group
    ]

    report = build_held_out_capacity_curve(
        training_events,
        evaluation_events,
        evaluation_groups=evaluation_groups,
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


def _run_heldout_breakdown(args: argparse.Namespace) -> int:
    manifest = args.collection_manifest.resolve()
    training, evaluations, excluded = load_collection_breakdown_inputs(
        manifest,
        phase=args.phase,
        evaluation_phase=args.eval_phase,
        max_layer=args.max_layer,
        exclude_failed=args.exclude_failed_shards,
    )
    report = build_heldout_breakdown(
        training,
        evaluations,
        capacity_per_layer=args.capacity,
        slot_bytes=args.slot_bytes,
        expert_transfer_ms=args.expert_transfer_ms,
    )
    report["collection_manifest"] = str(manifest)
    report["selection"] = {
        "training_phase": args.phase,
        "evaluation_phase": args.eval_phase or args.phase,
        "max_layer": args.max_layer,
    }
    report["excluded_shards"] = list(excluded)
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
    training_sources = [path.resolve() for path in (args.train or [])]
    evaluation_sources = [path.resolve() for path in (args.eval or [])]

    def selected(source: Path, phase: str):
        return [
            event
            for event in load_router_events(source)
            if event.phase == phase
            and (args.max_layer is None or event.layer_id <= args.max_layer)
        ]

    collection_manifest: Path | None = None
    evaluation_metadata: list[dict[str, str]] = []
    excluded_shards: list[dict[str, object]] = []
    if args.collection_manifest is not None:
        if training_sources or evaluation_sources:
            raise ValueError(
                "collection_manifest cannot be combined with train/eval paths"
            )
        collection_manifest = args.collection_manifest.resolve()
        training_loaded, evaluation_loaded, excluded = (
            load_collection_breakdown_inputs(
                collection_manifest,
                phase=args.train_phase,
                evaluation_phase=args.eval_phase,
                max_layer=args.max_layer,
                exclude_failed=args.exclude_failed_shards,
            )
        )
        training_events = list(training_loaded)
        evaluation_traces = [list(item.events) for item in evaluation_loaded]
        evaluation_sources = [
            Path(item.source_trace) for item in evaluation_loaded
        ]
        evaluation_metadata = [
            {
                "conversation_id": item.conversation_id,
                "split": item.split,
                "domain": item.domain,
                "source_trace": item.source_trace,
            }
            for item in evaluation_loaded
        ]
        excluded_shards = list(excluded)
    else:
        if not training_sources or not evaluation_sources:
            raise ValueError(
                "deadline-eval requires train/eval paths or collection_manifest"
            )
        if args.exclude_failed_shards:
            raise ValueError(
                "exclude_failed_shards requires collection_manifest"
            )
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
        transfer_backend=args.transfer_backend,
        window_backend=args.window_backend,
        transfer_statistic=args.transfer_statistic,
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
    if collection_manifest is not None:
        report["collection_manifest"] = str(collection_manifest)
        report["evaluation_metadata"] = evaluation_metadata
        report["excluded_shards"] = excluded_shards
    timing_sources: dict[str, str] = {}
    if args.transfer_source is not None:
        timing_sources["transfer"] = str(args.transfer_source.resolve())
    if args.window_source is not None:
        timing_sources["window"] = str(args.window_source.resolve())
    if timing_sources:
        report["timing_sources"] = timing_sources
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
    if args.command == "collect-pairs":
        return _run_collect_pairs(args)
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
    if args.command == "transfer-aggregate":
        return _run_transfer_aggregate(args)
    if args.command == "capacity-curve":
        return _run_capacity_curve(args)
    if args.command == "heldout-curve":
        return _run_heldout_curve(args)
    if args.command == "heldout-breakdown":
        return _run_heldout_breakdown(args)
    if args.command == "deadline-eval":
        return _run_deadline_eval(args)
    if args.command == "optimize":
        return _run_optimize(args)
    if args.command == "run":
        return _run_product_inference(args)
    if args.command == "serve":
        return _run_product_server(args)
    if args.command == "compare":
        print(json.dumps(compare_report(args.deployment), indent=2, sort_keys=True))
        return 0
    if args.command == "demo":
        report = replay_report()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
