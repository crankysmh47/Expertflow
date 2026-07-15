from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


TASKS = [
    ("python_palindrome", 128, "Write a Python function named is_palindrome(text) that ignores spaces, punctuation, and letter case. Return only the function inside one Python code block."),
    ("python_merge_intervals", 224, "Write a Python function named merge_intervals(intervals) that merges overlapping closed intervals. Return only the function inside one Python code block."),
    ("arithmetic_components", 48, "A warehouse has 37 boxes, each containing 48 components. Twelve components are damaged. How many usable components remain? Give only the final integer."),
    ("reasoning_request_rate", 48, "A service processes 120 requests per minute. Traffic increases by 25%, then 30 requests per minute are removed by caching. What is the final request rate? Give only the final integer."),
    ("structured_json", 96, 'Return valid JSON only with exactly these keys: "project", "status", and "risks". Set project to "ExpertFlow", status to "experimental", and risks to an array containing exactly "observer overhead" and "runtime complexity". Do not use Markdown code fences.'),
    ("translation_french", 96, 'Translate this sentence into French while preserving every fact: "The server restarted after the update, but no user data was lost." Return only the translation.'),
    ("reproducibility_bullets", 96, "Give exactly three bullet points explaining why reproducibility matters in systems research. Each bullet must contain six words or fewer."),
]


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--measurement-runner", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "observer"), required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "command-ledger.jsonl"
    probe = args.runtime.resolve() / "expertflow-router-probe.exe"
    prompt_template = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    environment = os.environ.copy()
    environment["PATH"] = str(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")) + os.pathsep + environment["PATH"]

    for task_id, n_predict, instruction in TASKS:
        run_dir = root / "runs" / task_id / args.mode
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = run_dir / "measurement.json"
        if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")).get("status") == "passed":
            continue
        tokens = run_dir / "tokens.json"
        trace = run_dir / "trace.jsonl"
        command = [str(probe), "-m", str(args.model.resolve()), "--tokens", str(tokens)]
        artifacts = ["--artifact", f"tokens={tokens}"]
        if args.mode == "observer":
            command += ["--trace", str(trace), "--trace-mode", "full"]
            artifacts += ["--artifact", f"trace={trace}"]
        else:
            command += ["--trace-mode", "disabled"]
        command += ["-n", str(n_predict), "-ngl", "10", "--threads", "12"]
        command.append(prompt_template.format(prompt=instruction))
        measured = [
            sys.executable, str(args.measurement_runner.resolve()),
            "--manifest", str(run_dir / "measurement.json"),
            "--cwd", str(args.runtime.resolve()), "--settle-seconds", "2",
            *artifacts, "--", *command,
        ]
        append_jsonl(ledger, {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": "command_start", "mode": args.mode, "task_id": task_id,
            "n_predict": n_predict, "command": measured,
        })
        result = subprocess.run(measured, env=environment, check=False)
        append_jsonl(ledger, {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": "command_end", "mode": args.mode, "task_id": task_id,
            "return_code": result.returncode,
        })
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
