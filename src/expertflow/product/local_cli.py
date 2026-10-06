"""Portable public local-model workflow; independent of research artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from expertflow.doctor import collect_doctor_report
from .paths import data_root
from .runtime import inspect_runtime, download_archive, install_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expertflow local", description="Set up, measure and run local GGUF models on your hardware.")
    sub = parser.add_subparsers(dest="action", required=True)
    doctor = sub.add_parser("doctor", help="Inspect hardware and optional runtime without loading a model.")
    doctor.add_argument("--runtime", type=Path)
    doctor.add_argument("--model", type=Path)
    doctor.add_argument("--dll-dir", type=Path, action="append", default=[])
    doctor.add_argument("--json", action="store_true")
    install = sub.add_parser("install-runtime", help="Explicitly download a checksum-verified llama.cpp archive.")
    install.add_argument("--url", required=True)
    install.add_argument("--sha256", required=True)
    install.add_argument("--destination", type=Path, required=True)
    install.add_argument("--json", action="store_true")
    setup = sub.add_parser("setup", help="Create a portable profile and verify a real baseline load.")
    setup.add_argument("--model", type=Path, required=True)
    setup.add_argument("--runtime", type=Path, required=True)
    setup.add_argument("--context", type=int, default=4096)
    setup.add_argument("--gpu-layers", type=int)
    setup.add_argument("--threads", type=int)
    setup.add_argument("--cpu-moe", action="store_true")
    setup.add_argument("--dll-dir", type=Path, action="append", default=[])
    setup.add_argument("--name")
    setup.add_argument("--output", type=Path)
    setup.add_argument("--no-probe", action="store_true", help="Save an explicitly unverified profile without a model process.")
    setup.add_argument("--health-timeout", type=int, default=180)
    setup.add_argument("--json", action="store_true")
    profiles = sub.add_parser("profiles", help="List, show or remove profiles; never delete model weights.")
    profiles.add_argument("operation", choices=["list", "show", "remove"])
    profiles.add_argument("--profile", type=Path)
    profiles.add_argument("--json", action="store_true")
    for action, help_text in [("run", "Stream a response or start an interactive local chat."),
                              ("serve", "Serve a profile on loopback for existing clients."),
                              ("bench", "Measure a baseline within a fixed process/wall budget."),
                              ("tune", "Search reviewed exact controls and independently confirm gains."),
                              ("report", "Inspect a profile and its archived measurements.")]:
        command = sub.add_parser(action, help=help_text)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--json", action="store_true")
        if action in {"run", "serve", "bench", "tune"}:
            command.add_argument("--output-dir", type=Path)
        if action in {"run", "serve"}:
            command.add_argument("--port", type=int, default=0 if action == "run" else 8080)
        if action == "run":
            command.add_argument("--prompt", help="One prompt; omit for interactive chat (/exit to finish).")
            command.add_argument("--predict", type=int, default=256)
        if action == "serve":
            command.add_argument("--duration", type=float, help="Optional bounded session duration in seconds.")
        if action in {"bench", "tune"}:
            command.add_argument("--budget-seconds", type=int, default=600)
        if action == "bench":
            command.add_argument("--predict", type=int, default=256)
            command.add_argument("--repeats", type=int, default=3)
            command.add_argument("--prompt-file", type=Path)
    for action in ["status", "stop"]:
        command=sub.add_parser(action,help="Inspect or stop a server after checking its saved process identity.")
        command.add_argument("--session-dir",type=Path,required=True)
        command.add_argument("--json",action="store_true")
    verify = sub.add_parser("verify-job", help="Reconstruct saved raw receipts without loading a model.")
    verify.add_argument("--job-dir", type=Path, required=True)
    verify.add_argument("--json", action="store_true")
    support = sub.add_parser("support", help="Explicitly export a redacted job summary; no uploads.")
    support.add_argument("--job-dir", type=Path, required=True)
    support.add_argument("--output", type=Path, required=True)
    support.add_argument("--json", action="store_true")
    return parser


def emit(report: dict, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return
    print(report["status"])
    if report.get("reason"):
        print(report["reason"])
    if report.get("hardware"):
        hardware = report["hardware"]
        print(f"RAM: {hardware['system_ram_bytes'] / 1024**3:.1f} GiB")
        for gpu in hardware["gpus"]:
            print(f"GPU: {gpu['name']} — {gpu['memory_free_mib']} MiB free / {gpu['memory_total_mib']} MiB")
        if hardware["gpu_error"]:
            print(f"GPU detection: {hardware['gpu_error']}")
    if report.get("runtime"):
        print(f"Runtime: {report['runtime']['version']}")
    if report.get("next_action"):
        print(report["next_action"])
    if report.get("profile_path"):
        print(f"Profile: {report['profile_path']}")
    if report.get("profiles"):
        for item in report["profiles"]:
            print(f"{item['name']}: {item['status']} — {item['path']}")
    if report.get("memory_estimate"):
        estimate = report["memory_estimate"]
        if estimate.get("resident_total_bytes") is not None:
            print(f"Estimated fully resident weights + F16 KV + reserve: {estimate['resident_total_bytes'] / 1024**3:.2f} GiB")
        print(estimate["uncertainty"])
    if report.get("measurement"):
        m = report["measurement"]
        print(f"Decode: {m['decode_tps']:.2f} tokens/s; TTFT: {m.get('ttft_seconds')} seconds")
    if report.get("job_directory"):
        print(f"Report: {Path(report['job_directory']) / 'report.json'}")
    if report.get("gain_pct") is not None:
        print(f"Held-out gain: {report['gain_pct']:+.2f}% (95% interval {report['ci95_pct']})")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "doctor":
            report = {"status": "INSPECTED", "hardware": collect_doctor_report(Path.cwd()),
                      "state_directory": str(data_root()),
                      "next_action": "Use local setup with a GGUF and the runtime bin directory."}
            if args.runtime:
                report["runtime"] = inspect_runtime(args.runtime, dll_dirs=args.dll_dir)
            if args.model:
                if not args.model.is_file():
                    raise ValueError("Model file is missing; correct --model.")
                report["model"] = {"path": str(args.model.resolve()), "size_bytes": args.model.stat().st_size}
        elif args.action in {"verify-job", "support"}:
            from .reports import verify_job, support_summary
            if args.action == "verify-job":
                report = verify_job(args.job_dir)
            else:
                report = support_summary(args.job_dir)
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(report, indent=2) + "\n")
                report = {"status":"EXPORTED", "reason":"Redacted summary saved; review it before sharing.", "output":str(args.output)}
        elif args.action == "install-runtime":
            archive = args.destination.parent / (args.destination.name + ".download")
            download_archive(args.url, archive, args.sha256)
            directory = install_archive(archive, args.destination)
            report = {"status": "INSTALLED", "directory": str(directory),
                      "next_action": f"Run local doctor --runtime \"{directory}\" to check dependencies."}
        elif args.action in {"status", "stop"}:
            from .service_control import process_state, stop_session
            report = process_state(args.session_dir) if args.action == "status" else stop_session(args.session_dir)
        elif args.action == "setup":
            from .setup import setup_profile
            import uuid
            output = args.output or data_root() / "profiles" / (uuid.uuid4().hex + ".json")
            profile = setup_profile(args.model, args.runtime, output, context=args.context,
                gpu_layers=args.gpu_layers, threads=args.threads, cpu_moe=args.cpu_moe,
                dll_dirs=args.dll_dir, name=args.name, probe=not args.no_probe,
                health_timeout=args.health_timeout)
            report = {"status": profile["status"], "profile_path": str(output.resolve()),
                "memory_estimate": profile["memory_estimate"], "load_verified": profile["load_verified"],
                "next_action": f'Run: expertflow local run --profile "{output.resolve()}"'}
        elif args.action == "profiles":
            from .profiles import load_profile
            if args.operation == "list":
                items = []
                for path in sorted((data_root() / "profiles").glob("*.json")):
                    try:
                        p = load_profile(path)
                        items.append({"name": p["name"], "status": p["status"], "path": str(path)})
                    except ValueError:
                        items.append({"name": path.stem, "status": "INVALID-PROFILE", "path": str(path)})
                report = {"status": "INSPECTED", "profiles": items}
            else:
                if args.profile is None:
                    raise ValueError("--profile is required for show/remove.")
                p = load_profile(args.profile)
                if args.operation == "remove":
                    args.profile.unlink()
                    report = {"status": "REMOVED", "reason": "Profile removed; model and runtime files retained."}
                else:
                    report = {"status": p["status"], "profile": p, "profile_path": str(args.profile)}
        elif args.action in {"run", "serve", "bench", "tune", "report"}:
            from .profiles import load_profile
            import uuid
            profile = load_profile(args.profile)
            if args.action == "report":
                report = {"status": profile["status"], "profile": profile,
                          "memory_estimate": profile["memory_estimate"],
                          "reason": "Saved profile/receipts; archived measurements are not a fresh benchmark.",
                          "next_action": "Use local bench for a fresh measurement, or local run/serve to launch with identity checks."}
            elif args.action in {"bench", "tune"}:
                output = args.output_dir or data_root() / "jobs" / uuid.uuid4().hex
                if args.action == "bench":
                    from .benchmark import bench_profile
                    prompts = [args.prompt_file.read_text(encoding="utf-8")] if args.prompt_file else None
                    report = bench_profile(profile, output, predict=args.predict, repeats=args.repeats,
                                           budget_seconds=args.budget_seconds, prompts=prompts)
                else:
                    from .tuning import tune_profile
                    report = tune_profile(profile, output, budget_seconds=args.budget_seconds)
                report["job_directory"] = str(output.resolve())
                report["next_action"] = f'Use selected-profile.json only for VERIFIED-IMPROVEMENT. Otherwise keep: "{args.profile}"'
                emit(report, json_output=args.json)
                return 0 if report["status"] in {"MEASURED", "VERIFIED-IMPROVEMENT", "NO-MEASURABLE-GAIN"} else 2
            else:
                from .server import ServerSession
                import time
                output = args.output_dir or data_root() / "sessions" / uuid.uuid4().hex
                if args.action == "run" and args.json and not args.prompt:
                    raise ValueError("JSON run needs --prompt; use interactive run without --json.")
                if args.action == "serve" and args.duration is not None and not 0 < args.duration <= 86400:
                    raise ValueError("Session duration must be between 0 and 86400 seconds.")
                if args.action == "run" and not 1 <= args.predict <= 65536:
                    raise ValueError("Predict tokens must be between 1 and 65536.")
                if args.action == "run" and args.prompt is not None and not args.prompt.strip():
                    raise ValueError("One-shot prompt must contain non-whitespace text.")
                with ServerSession(profile, log_dir=output, port=args.port) as server:
                    if args.action == "serve":
                        emit({"status":"SERVING", "reason":f"Local API: {server.url}/v1 — web UI: {server.url}",
                              "next_action":f'Press Ctrl+C, or run: expertflow local stop --session-dir "{output}"', "session_directory":str(output)}, json_output=args.json)
                        deadline=time.monotonic()+args.duration if args.duration else float('inf')
                        while server.child.process.poll() is None and time.monotonic() < deadline:
                            time.sleep(0.2)
                        if server.child.process.poll() not in {None,0}:
                            raise ValueError(f"Server exited unexpectedly; inspect {output / 'server.log'}.")
                        return 0
                    messages, answer = [], ""
                    while True:
                        try:
                            prompt = args.prompt if args.prompt is not None else input("You: ")
                        except EOFError:
                            break
                        if prompt.strip() in {"/exit", "/quit"}:
                            break
                        if not prompt.strip():
                            continue
                        messages.append({"role":"user", "content":prompt})
                        pieces = []
                        for chunk in server.chat(messages, predict=args.predict):
                            pieces.append(chunk)
                            if not args.json:
                                print(chunk, end="", flush=True)
                        answer = "".join(pieces)
                        messages.append({"role":"assistant", "content":answer})
                        if not args.json:
                            print()
                        if args.prompt is not None:
                            break
                if args.json:
                    emit({"status":"COMPLETED", "response":answer, "log_directory":str(output)}, json_output=True)
                return 0
        emit(report, json_output=args.json)
        return 0
    except KeyboardInterrupt:
        emit({"status":"STOPPED", "reason":"Cancelled; owned server stopped and diagnostics retained."}, json_output=args.json)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        emit({"status": "UNSUPPORTED", "reason": str(error),
              "next_action": "Correct the reported input/dependency. Failed launch logs are retained; use a fresh output directory."}, json_output=args.json)
        return 2


if __name__ == "__main__":
    sys.exit(main())
