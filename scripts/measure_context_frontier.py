from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading

from scripts.benchmark_product_server import (
    STATIC_LAYERS,
    _owned_vram_monitor,
    _request,
    _wait_health,
)


def choose_context_profile(rows: list[dict[str, object]], *, total_vram_mib: float, margin_mib: float) -> dict[str, object]:
    eligible = [
        row for row in rows
        if row["mode"] == "expertflow"
        and row["status"] == "pass"
        and int(row["processed_tokens"]) > 0
        and float(row["peak_process_owned_vram_mib"]) <= total_vram_mib - margin_mib
    ]
    if not eligible:
        raise ValueError("no measured context preserves the VRAM margin")
    return max(eligible, key=lambda row: int(row["allocated_context"]))


def run_context(*, mode: str, server: Path, model: Path, context: int, port: int, output: Path) -> dict[str, object]:
    run_dir = output / f"{mode}-ctx{context}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = run_dir / "stdout.log", run_dir / "stderr.log"
    env = os.environ.copy()
    cuda_bin = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")) / "bin"
    env["PATH"] = f"{cuda_bin};{server.parent};{env.get('PATH', '')}"
    for name in ("LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER", "LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE", "GGML_CUDA_DISABLE_GRAPHS"):
        env.pop(name, None)
    if mode == "expertflow":
        env["LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER"] = STATIC_LAYERS
        env["LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE"] = "1"
    command = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", str(context), "-np", "1", "-ngl", "99", "--threads", "12", "--cpu-moe", "--metrics"]
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=server.parent, env=env, stdout=stdout, stderr=stderr)
        stop = threading.Event()
        memory: dict[str, float] = {}
        monitor = threading.Thread(target=_owned_vram_monitor, args=(process.pid, stop, memory), daemon=True)
        monitor.start()
        status, error, response = "pass", None, None
        try:
            _wait_health(port, process)
            prompt = (" cache locality evidence" * 128).strip()
            response = _request(port, prompt, 32)
        except Exception as exc:
            status, error = "oom_or_runtime_failure", repr(exc)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stop.set()
            monitor.join(timeout=10)
    row = {
        "mode": mode,
        "allocated_context": context,
        "status": status,
        "processed_tokens": 0 if response is None else int(response["prompt_tokens"]) + int(response["generated_tokens"]),
        "prompt_tokens": 0 if response is None else response["prompt_tokens"],
        "generated_tokens": 0 if response is None else response["generated_tokens"],
        "prompt_tps": None if response is None else response["prompt_tps"],
        "decode_tps": None if response is None else response["predicted_tps"],
        "ttft_seconds": None if response is None else response["ttft_seconds"],
        "peak_process_owned_vram_mib": memory.get("peak_mib", 0.0),
        "response_sha256": None if response is None else response["response_sha256"],
        "command": command,
        "error": error,
    }
    (run_dir / "measurement.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure live allocated context with a bounded processed-token proof.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock-server", type=Path)
    parser.add_argument("--expertflow-server", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--context", type=int, action="append", default=[])
    parser.add_argument("--port", type=int, default=8141)
    args = parser.parse_args()
    if args.input:
        rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    else:
        if not (args.stock_server and args.expertflow_server and args.model):
            parser.error("live mode requires --stock-server, --expertflow-server, and --model")
        rows = []
        contexts = args.context or [8192, 16384, 32768, 65536, 131072]
        args.output.mkdir(parents=True, exist_ok=True)
        for context in contexts:
            for offset, (mode, server) in enumerate((("stock", args.stock_server), ("expertflow", args.expertflow_server))):
                row = run_context(mode=mode, server=server.resolve(), model=args.model.resolve(), context=context, port=args.port + offset, output=args.output)
                rows.append(row)
                (args.output / "checkpoint.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
            if any(row["status"] != "pass" for row in rows[-2:]):
                break
    selected = choose_context_profile(rows, total_vram_mib=16311, margin_mib=512)
    destination = args.output if args.output.suffix else args.output / "context-results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema_version": "1.0.0",
        "measurement_kind": "live_context_frontier",
        "rows": rows,
        "selected": selected,
        "notice": "Allocated context was live and processed the recorded token count; unfilled capacity is not reported as processed tokens.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
