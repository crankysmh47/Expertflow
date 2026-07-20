from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


STATIC_LAYERS = "0,1,2,3,4,5,6,7,8,9,15,20"
PROMPTS = (
    "Explain why a bounded cache can improve locality in four concise paragraphs.",
    "Write a Python function that validates a SHA-256 manifest and explain its edge cases.",
    "Translate 'Reliable systems make failures explicit' into French, Urdu, and Japanese.",
    "Return JSON with three practical tests for a local OpenAI-compatible inference server.",
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def aggregate_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for run in runs:
        groups.setdefault((str(run["mode"]), int(run["parallel_slots"])), []).append(run)
    rows = []
    for (mode, slots), items in sorted(groups.items()):
        tps = [float(item["aggregate_generated_tps"]) for item in items]
        latencies = [float(value) for item in items for value in item["request_latency_seconds"]]
        ttft = [float(value) for item in items for value in item["ttft_seconds"] if value is not None]
        prompt_tps = [float(value) for item in items for value in item.get("prompt_tps", []) if value is not None]
        rows.append({
            "mode": mode,
            "parallel_slots": slots,
            "repetitions": len(items),
            "aggregate_generated_tps_values": tps,
            "aggregate_generated_tps_mean": statistics.mean(tps),
            "aggregate_generated_tps_sample_variance": statistics.variance(tps) if len(tps) > 1 else 0.0,
            "per_request_tps_mean": statistics.mean(float(value) for item in items for value in item["per_request_tps"]),
            "prompt_tps_mean": statistics.mean(prompt_tps) if prompt_tps else None,
            "median_request_latency_seconds": statistics.median(latencies),
            "p95_request_latency_seconds": _percentile(latencies, 0.95),
            "median_ttft_seconds": statistics.median(ttft) if ttft else None,
            "p95_ttft_seconds": _percentile(ttft, 0.95) if ttft else None,
            "peak_process_owned_vram_mib": max(float(item["peak_process_owned_vram_mib"]) for item in items),
            "completed_requests": sum(int(item["completed_requests"]) for item in items),
            "error_count": sum(int(item["errors"]) for item in items),
        })
    return rows


def choose_profile(rows: list[dict[str, object]], *, total_vram_mib: float, margin_mib: float) -> dict[str, object]:
    eligible = [row for row in rows if row["mode"] == "expertflow" and int(row["error_count"]) == 0 and float(row["peak_process_owned_vram_mib"]) <= total_vram_mib - margin_mib]
    if not eligible:
        raise ValueError("no stable ExpertFlow batching profile preserves the VRAM margin")
    eligible.sort(key=lambda row: (float(row["aggregate_generated_tps_mean"]), -float(row["p95_request_latency_seconds"])), reverse=True)
    return eligible[0]


def _wait_health(port: int, process: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load with code {process.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(1)
    raise TimeoutError("server health timeout")


def _request(port: int, prompt: str, n_predict: int) -> dict[str, object]:
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "seed": 42, "cache_prompt": False, "stream": True}).encode()
    request = Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    content: list[str] = []
    final: dict[str, object] = {}
    with urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            piece = event.get("content") or ""
            if piece and first is None:
                first = time.perf_counter()
            content.append(piece)
            if event.get("stop"):
                final = event
    end = time.perf_counter()
    timings = final.get("timings") or {}
    generated = int(timings.get("predicted_n") or final.get("tokens_predicted") or n_predict)
    return {
        "generated_tokens": generated,
        "prompt_tokens": int(timings.get("prompt_n") or 0),
        "latency_seconds": end - start,
        "ttft_seconds": None if first is None else first - start,
        "predicted_tps": timings.get("predicted_per_second"),
        "prompt_tps": timings.get("prompt_per_second"),
        "response_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
    }


def _owned_vram_monitor(pid: int, stop: threading.Event, result: dict[str, float]) -> None:
    peak = 0.0
    command = "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.Status -eq 0 -and $_.InstanceName -match 'pid_%d(_|$)' } | Measure-Object CookedValue -Sum | Select-Object -ExpandProperty Sum" % pid
    while not stop.wait(0.5):
        completed = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
        try:
            peak = max(peak, float(completed.stdout.strip() or 0) / (1024 * 1024))
        except ValueError:
            pass
    result["peak_mib"] = peak


def run_once(*, mode: str, server: Path, model: Path, slots: int, repetition: int, port: int, n_predict: int, output: Path) -> dict[str, object]:
    run_dir = output / f"{mode}-slots{slots}-rep{repetition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout = (run_dir / "stdout.log").open("wb")
    stderr = (run_dir / "stderr.log").open("wb")
    env = os.environ.copy()
    cuda_bin = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")) / "bin"
    env["PATH"] = f"{cuda_bin};{server.parent};{env.get('PATH', '')}"
    for name in ("LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER", "LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE", "GGML_CUDA_DISABLE_GRAPHS"):
        env.pop(name, None)
    if mode == "expertflow":
        env["LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER"] = STATIC_LAYERS
        env["LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE"] = "1"
    command = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", str(2048 * slots), "-np", str(slots), "-ngl", "99", "--threads", "12", "--cpu-moe", "--metrics", "--cont-batching"]
    process = subprocess.Popen(command, cwd=server.parent, env=env, stdout=stdout, stderr=stderr)
    stop = threading.Event()
    memory: dict[str, float] = {}
    monitor = threading.Thread(target=_owned_vram_monitor, args=(process.pid, stop, memory), daemon=True)
    monitor.start()
    errors: list[str] = []
    responses: list[dict[str, object]] = []
    try:
        _wait_health(port, process)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=slots) as pool:
            futures = [pool.submit(_request, port, PROMPTS[index % len(PROMPTS)], n_predict) for index in range(slots)]
            for future in futures:
                try:
                    responses.append(future.result())
                except Exception as error:
                    errors.append(repr(error))
        wall = time.perf_counter() - start
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        stop.set(); monitor.join(timeout=10)
        stdout.close(); stderr.close()
    generated = sum(int(item["generated_tokens"]) for item in responses)
    row = {
        "mode": mode, "parallel_slots": slots, "repetition": repetition,
        "command": command, "generated_tokens": generated, "wall_seconds": wall,
        "aggregate_generated_tps": generated / wall if wall else 0.0,
        "per_request_tps": [float(item["predicted_tps"] or (int(item["generated_tokens"]) / float(item["latency_seconds"]))) for item in responses],
        "prompt_tps": [item["prompt_tps"] for item in responses],
        "request_latency_seconds": [item["latency_seconds"] for item in responses],
        "ttft_seconds": [item["ttft_seconds"] for item in responses],
        "peak_process_owned_vram_mib": memory.get("peak_mib", 0.0),
        "completed_requests": len(responses), "errors": len(errors), "error_messages": errors,
        "response_sha256": [item["response_sha256"] for item in responses],
    }
    (run_dir / "measurement.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-server", type=Path, required=True)
    parser.add_argument("--expertflow-server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    slots = args.slots or [1, 2, 4]
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for slot_count in slots:
        for repetition in range(1, args.repetitions + 1):
            for offset, (mode, server) in enumerate((("stock", args.stock_server), ("expertflow", args.expertflow_server))):
                row = run_once(mode=mode, server=server.resolve(), model=args.model.resolve(), slots=slot_count, repetition=repetition, port=args.port + offset, n_predict=args.n_predict, output=args.output)
                runs.append(row)
                (args.output / "checkpoint.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    summary = aggregate_runs(runs)
    selected = choose_profile(summary, total_vram_mib=16311, margin_mib=512)
    result = {"schema_version": "1.0.0", "measurement_kind": "live_continuous_batching", "runs": runs, "summary": summary, "selected": selected}
    (args.output / "batching-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
