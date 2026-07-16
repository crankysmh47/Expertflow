from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from expertflow.predictor.live_shadow import validate_token_and_router_parity
from expertflow.predictor.temporal_sidecar_analysis import (
    compare_t2_pair,
    summarize_t2_run,
)
from scripts.run_p1_live_shadow_suite import FOCUSED, prompt


ENVIRONMENT_NAMES = (
    "EXPERTFLOW_LIVE_CACHE",
    "EXPERTFLOW_LIVE_CACHE_MODE",
    "EXPERTFLOW_LIVE_CACHE_LAYER",
    "EXPERTFLOW_LIVE_CACHE_LAYERS",
    "EXPERTFLOW_LIVE_CACHE_LOG",
    "EXPERTFLOW_LIVE_CACHE_LOG_DETAIL",
    "EXPERTFLOW_PREDICTOR_SHADOW",
    "EXPERTFLOW_PREDICTOR_ARTIFACT",
    "EXPERTFLOW_PREDICTOR_SHADOW_LOG",
    "EXPERTFLOW_PREDICTOR_RUN_ID",
    "EXPERTFLOW_P2_MODE",
    "EXPERTFLOW_P2_LOG",
    "EXPERTFLOW_TEMPORAL_SHADOW",
    "EXPERTFLOW_TEMPORAL_ARTIFACT",
    "EXPERTFLOW_TEMPORAL_SHADOW_LOG",
    "EXPERTFLOW_TEMPORAL_RUN_ID",
    "EXPERTFLOW_T2_TEMPORAL_SIDECAR",
    "EXPERTFLOW_T2_LOG",
    "EXPERTFLOW_T2_RUN_ID",
)


def _append(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _add_memory(summary: dict[str, object], measurement_path: Path) -> None:
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    memory = measurement["memory"]
    peak = memory["gpu_peak_used_mib"]
    summary["gpu_peak_used_mib"] = max(int(value) for value in peak.values())
    summary["process_peak_working_set_bytes"] = int(
        memory["process_peak"]["peak_working_set_bytes"]
    )
    summary["process_peak_pagefile_bytes"] = int(
        memory["process_peak"]["peak_pagefile_bytes"]
    )
    summary["duration_seconds"] = float(measurement["duration_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--measurement-runner", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--n-predict", type=int, default=16)
    args = parser.parse_args()
    if args.repetitions < 4 or args.n_predict < 2:
        parser.error(
            "T2 requires one warmup plus three measured repetitions "
            "and at least two decode tokens"
        )

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve()
    probe = runtime / "expertflow-router-probe.exe"
    ledger = root / "command-ledger.jsonl"
    environment_base = os.environ.copy()
    environment_base["PATH"] = (
        str(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"))
        + os.pathsep
        + environment_base["PATH"]
    )
    for name in ENVIRONMENT_NAMES:
        environment_base.pop(name, None)

    for task_id, instruction in FOCUSED.items():
        for repetition in range(args.repetitions):
            pair = root / task_id / f"rep-{repetition:02d}"
            for mode in ("reactive", "t2"):
                run = pair / mode
                run.mkdir(parents=True, exist_ok=True)
                tokens = run / "tokens.json"
                trace = run / "trace.jsonl"
                performance = run / "performance.json"
                measurement = run / "measurement.json"
                cache = run / "cache.jsonl"
                temporal = run / "temporal.jsonl"
                sidecar = run / "t2.jsonl"
                command = [
                    str(probe),
                    "-m",
                    str(args.model.resolve()),
                    "--tokens",
                    str(tokens),
                    "--performance",
                    str(performance),
                    "--trace",
                    str(trace),
                    "--trace-mode",
                    "full",
                    "-n",
                    str(args.n_predict),
                    "-ngl",
                    "10",
                    "--threads",
                    "12",
                    prompt(instruction),
                ]
                measured = [
                    sys.executable,
                    str(args.measurement_runner.resolve()),
                    "--manifest",
                    str(measurement),
                    "--cwd",
                    str(runtime),
                    "--sample-interval",
                    "0.1",
                    "--settle-seconds",
                    "0.5",
                    "--artifact",
                    f"tokens={tokens}",
                    "--artifact",
                    f"trace={trace}",
                    "--artifact",
                    f"performance={performance}",
                    "--artifact",
                    f"cache={cache}",
                ]
                environment = environment_base.copy()
                environment.update(
                    {
                        "EXPERTFLOW_LIVE_CACHE": "1",
                        "EXPERTFLOW_LIVE_CACHE_MODE": "blocking",
                        "EXPERTFLOW_LIVE_CACHE_LAYER": "24",
                        "EXPERTFLOW_LIVE_CACHE_LOG": str(cache),
                    }
                )
                if mode == "t2":
                    run_id = f"t2-{task_id}-{repetition:02d}"
                    environment.update(
                        {
                            "EXPERTFLOW_TEMPORAL_SHADOW": "1",
                            "EXPERTFLOW_TEMPORAL_ARTIFACT": str(
                                args.artifact.resolve()
                            ),
                            "EXPERTFLOW_TEMPORAL_SHADOW_LOG": str(temporal),
                            "EXPERTFLOW_TEMPORAL_RUN_ID": run_id,
                            "EXPERTFLOW_T2_TEMPORAL_SIDECAR": "1",
                            "EXPERTFLOW_T2_LOG": str(sidecar),
                            "EXPERTFLOW_T2_RUN_ID": run_id,
                        }
                    )
                    measured.extend(
                        [
                            "--artifact",
                            f"temporal={temporal}",
                            "--artifact",
                            f"t2={sidecar}",
                        ]
                    )
                measured.extend(["--", *command])
                _append(
                    ledger,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "event": "command_start",
                        "task_id": task_id,
                        "repetition": repetition,
                        "mode": mode,
                        "command": measured,
                        "environment": {
                            key: environment[key]
                            for key in ENVIRONMENT_NAMES
                            if key in environment
                        },
                    },
                )
                result = subprocess.run(measured, env=environment, check=False)
                _append(
                    ledger,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "event": "command_end",
                        "task_id": task_id,
                        "repetition": repetition,
                        "mode": mode,
                        "return_code": result.returncode,
                    },
                )
                if result.returncode != 0:
                    return result.returncode

            validate_token_and_router_parity(
                pair / "reactive/tokens.json",
                pair / "t2/tokens.json",
                pair / "reactive/trace.jsonl",
                pair / "t2/trace.jsonl",
            )
            reactive_summary = summarize_t2_run(
                pair / "reactive/performance.json",
                pair / "reactive/cache.jsonl",
                None,
            )
            t2_summary = summarize_t2_run(
                pair / "t2/performance.json",
                pair / "t2/cache.jsonl",
                pair / "t2/t2.jsonl",
            )
            _add_memory(reactive_summary, pair / "reactive/measurement.json")
            _add_memory(t2_summary, pair / "t2/measurement.json")
            summary = {
                "task_id": task_id,
                "repetition": repetition,
                "warmup": repetition == 0,
                "token_parity": "exact",
                "router_parity": "exact",
                "reactive": reactive_summary,
                "t2": t2_summary,
                "comparison": compare_t2_pair(
                    reactive_summary, t2_summary
                ),
            }
            (pair / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _append(
                ledger,
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "event": "pair_validated",
                    "task_id": task_id,
                    "repetition": repetition,
                    "token_parity": "exact",
                    "router_parity": "exact",
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
