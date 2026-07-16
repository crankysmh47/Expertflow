from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from expertflow.predictor.live_shadow import (
    load_shadow_log,
    summarize_shadow_records,
    validate_offline_equivalence,
    validate_token_and_router_parity,
)
from expertflow.predictor.runtime_artifact import parse_runtime_artifact


FOCUSED = {
    "general": (
        "Explain in three concise bullet points why reproducible measurements "
        "matter when optimizing local AI inference."
    ),
    "code": (
        "Write a Python function named merge_intervals(intervals) that merges "
        "overlapping closed intervals. Return only the function inside one "
        "Python code block."
    ),
    "translation": (
        'Translate this sentence into French while preserving every fact: '
        '"The server restarted after the update, but no user data was lost." '
        "Return only the translation."
    ),
}

SMOKE = {
    "python_palindrome": (
        "Write a Python function named is_palindrome(text) that ignores spaces, "
        "punctuation, and letter case. Return only the function inside one "
        "Python code block."
    ),
    "python_merge_intervals": FOCUSED["code"],
    "arithmetic_components": (
        "A warehouse has 37 boxes, each containing 48 components. Twelve "
        "components are damaged. How many usable components remain? Give only "
        "the final integer."
    ),
    "reasoning_request_rate": (
        "A service processes 120 requests per minute. Traffic increases by "
        "25%, then 30 requests per minute are removed by caching. What is the "
        "final request rate? Give only the final integer."
    ),
    "structured_json": (
        'Return valid JSON only with exactly these keys: "project", "status", '
        'and "risks". Set project to "ExpertFlow", status to "experimental", '
        'and risks to an array containing exactly "observer overhead" and '
        '"runtime complexity". Do not use Markdown code fences.'
    ),
    "translation_french": FOCUSED["translation"],
    "reproducibility_bullets": (
        "Give exactly three bullet points explaining why reproducibility "
        "matters in systems research. Each bullet must contain six words or fewer."
    ),
}


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")


def prompt(instruction: str) -> str:
    return (
        "<start_of_turn>user\n"
        + instruction
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--measurement-runner", type=Path, required=True)
    parser.add_argument("--suite", choices=("focused", "smoke"), required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=16)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.n_predict <= 0:
        parser.error("repetitions and n-predict must be positive")

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "command-ledger.jsonl"
    runtime = args.runtime.resolve()
    probe = runtime / "expertflow-router-probe.exe"
    artifact = parse_runtime_artifact(args.artifact.read_bytes())
    tasks = FOCUSED if args.suite == "focused" else SMOKE

    base_environment = os.environ.copy()
    cuda = Path(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    )
    base_environment["PATH"] = str(cuda) + os.pathsep + base_environment["PATH"]
    for name in (
        "EXPERTFLOW_LIVE_CACHE",
        "EXPERTFLOW_LIVE_CACHE_MODE",
        "EXPERTFLOW_LIVE_CACHE_LOG",
        "EXPERTFLOW_PREDICTOR_SHADOW",
        "EXPERTFLOW_PREDICTOR_ARTIFACT",
        "EXPERTFLOW_PREDICTOR_SHADOW_LOG",
        "EXPERTFLOW_PREDICTOR_RUN_ID",
    ):
        base_environment.pop(name, None)

    for task_id, instruction in tasks.items():
        for repetition in range(args.repetitions):
            pair_dir = root / task_id / f"rep-{repetition:02d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            paths: dict[str, dict[str, Path]] = {}
            for mode in ("disabled", "shadow"):
                run_dir = pair_dir / mode
                run_dir.mkdir(parents=True, exist_ok=True)
                paths[mode] = {
                    "tokens": run_dir / "tokens.json",
                    "trace": run_dir / "trace.jsonl",
                    "performance": run_dir / "performance.json",
                    "measurement": run_dir / "measurement.json",
                    "shadow": run_dir / "shadow.jsonl",
                }
                command = [
                    str(probe),
                    "-m",
                    str(args.model.resolve()),
                    "--tokens",
                    str(paths[mode]["tokens"]),
                    "--performance",
                    str(paths[mode]["performance"]),
                    "--trace",
                    str(paths[mode]["trace"]),
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
                    str(paths[mode]["measurement"]),
                    "--cwd",
                    str(runtime),
                    "--sample-interval",
                    "0.1",
                    "--settle-seconds",
                    "0.5",
                    "--artifact",
                    f"tokens={paths[mode]['tokens']}",
                    "--artifact",
                    f"trace={paths[mode]['trace']}",
                    "--artifact",
                    f"performance={paths[mode]['performance']}",
                ]
                environment = base_environment.copy()
                if mode == "shadow":
                    environment.update(
                        {
                            "EXPERTFLOW_PREDICTOR_SHADOW": "1",
                            "EXPERTFLOW_PREDICTOR_ARTIFACT": str(
                                args.artifact.resolve()
                            ),
                            "EXPERTFLOW_PREDICTOR_SHADOW_LOG": str(
                                paths[mode]["shadow"]
                            ),
                            "EXPERTFLOW_PREDICTOR_RUN_ID": (
                                f"{args.suite}-{task_id}-{repetition:02d}"
                            ),
                        }
                    )
                    measured += [
                        "--artifact",
                        f"shadow={paths[mode]['shadow']}",
                    ]
                measured += ["--", *command]
                append_jsonl(
                    ledger,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "event": "command_start",
                        "suite": args.suite,
                        "task_id": task_id,
                        "repetition": repetition,
                        "mode": mode,
                        "command": measured,
                    },
                )
                result = subprocess.run(measured, env=environment, check=False)
                append_jsonl(
                    ledger,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "event": "command_end",
                        "suite": args.suite,
                        "task_id": task_id,
                        "repetition": repetition,
                        "mode": mode,
                        "return_code": result.returncode,
                    },
                )
                if result.returncode != 0:
                    return result.returncode

            validate_token_and_router_parity(
                paths["disabled"]["tokens"],
                paths["shadow"]["tokens"],
                paths["disabled"]["trace"],
                paths["shadow"]["trace"],
            )
            records, runtime_summary = load_shadow_log(paths["shadow"]["shadow"])
            validate_offline_equivalence(artifact, records)
            pair_summary = summarize_shadow_records(records, runtime_summary)
            pair_summary.update(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "repetition": repetition,
                    "token_parity": "exact",
                    "router_parity": "exact",
                    "offline_live_equivalence": "exact",
                }
            )
            (pair_dir / "summary.json").write_text(
                json.dumps(pair_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            append_jsonl(
                ledger,
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "event": "pair_validated",
                    **pair_summary,
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
