from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from expertflow.predictor.live_shadow import validate_token_and_router_parity
from expertflow.predictor.temporal_live_shadow import (
    load_temporal_shadow_log,
    summarize_temporal_shadow,
    validate_temporal_offline_equivalence,
)
from expertflow.predictor.temporal_runtime_artifact import (
    parse_temporal_runtime_artifact,
)
from scripts.run_p1_live_shadow_suite import FOCUSED, prompt


def _append(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


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
        parser.error("T1 requires one warmup plus three measured repetitions and at least two decode tokens")

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve()
    probe = runtime / "expertflow-router-probe.exe"
    artifact = parse_temporal_runtime_artifact(args.artifact.read_bytes())
    ledger = root / "command-ledger.jsonl"
    environment_base = os.environ.copy()
    environment_base["PATH"] = (
        str(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"))
        + os.pathsep + environment_base["PATH"]
    )
    for name in (
        "EXPERTFLOW_LIVE_CACHE",
        "EXPERTFLOW_LIVE_CACHE_MODE",
        "EXPERTFLOW_LIVE_CACHE_LOG",
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
    ):
        environment_base.pop(name, None)

    for task_id, instruction in FOCUSED.items():
        for repetition in range(args.repetitions):
            pair = root / task_id / f"rep-{repetition:02d}"
            for mode in ("disabled", "temporal"):
                run = pair / mode
                run.mkdir(parents=True, exist_ok=True)
                tokens = run / "tokens.json"
                trace = run / "trace.jsonl"
                performance = run / "performance.json"
                measurement = run / "measurement.json"
                shadow = run / "temporal-shadow.jsonl"
                command = [
                    str(probe), "-m", str(args.model.resolve()),
                    "--tokens", str(tokens), "--performance", str(performance),
                    "--trace", str(trace), "--trace-mode", "full",
                    "-n", str(args.n_predict), "-ngl", "10", "--threads", "12",
                    prompt(instruction),
                ]
                measured = [
                    sys.executable, str(args.measurement_runner.resolve()),
                    "--manifest", str(measurement), "--cwd", str(runtime),
                    "--sample-interval", "0.1", "--settle-seconds", "0.5",
                    "--artifact", f"tokens={tokens}",
                    "--artifact", f"trace={trace}",
                    "--artifact", f"performance={performance}",
                ]
                environment = environment_base.copy()
                if mode == "temporal":
                    environment.update({
                        "EXPERTFLOW_TEMPORAL_SHADOW": "1",
                        "EXPERTFLOW_TEMPORAL_ARTIFACT": str(args.artifact.resolve()),
                        "EXPERTFLOW_TEMPORAL_SHADOW_LOG": str(shadow),
                        "EXPERTFLOW_TEMPORAL_RUN_ID": f"t1-{task_id}-{repetition:02d}",
                    })
                    measured += ["--artifact", f"temporal_shadow={shadow}"]
                measured += ["--", *command]
                _append(ledger, {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "event": "command_start", "task_id": task_id,
                    "repetition": repetition, "mode": mode, "command": measured,
                })
                result = subprocess.run(measured, env=environment, check=False)
                _append(ledger, {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "event": "command_end", "task_id": task_id,
                    "repetition": repetition, "mode": mode,
                    "return_code": result.returncode,
                })
                if result.returncode != 0:
                    return result.returncode
            validate_token_and_router_parity(
                pair / "disabled/tokens.json",
                pair / "temporal/tokens.json",
                pair / "disabled/trace.jsonl",
                pair / "temporal/trace.jsonl",
            )
            records, runtime_summary = load_temporal_shadow_log(
                pair / "temporal/temporal-shadow.jsonl"
            )
            validate_temporal_offline_equivalence(artifact, records)
            summary = summarize_temporal_shadow(records, runtime_summary)
            summary.update({
                "task_id": task_id,
                "repetition": repetition,
                "warmup": repetition == 0,
                "token_parity": "exact",
                "router_parity": "exact",
                "offline_live_equivalence": "exact",
            })
            (pair / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _append(ledger, {
                "at": datetime.now(timezone.utc).isoformat(),
                "event": "pair_validated", **summary,
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

