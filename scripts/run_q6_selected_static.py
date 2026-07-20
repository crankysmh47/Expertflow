from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request


STATIC_LAYERS = "0,1,15,20"
MODEL_SHA256 = "089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pair_order(count: int) -> list[tuple[str, str]]:
    if count <= 0:
        raise ValueError("pair count must be positive")
    return [("off", "on") if index % 2 == 0 else ("on", "off") for index in range(count)]


def build_environment(inherited: dict[str, str], mode: str) -> dict[str, str]:
    if mode not in {"off", "on"}:
        raise ValueError(f"unknown mode: {mode}")
    result = dict(inherited)
    result.pop("LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER", None)
    result.pop("LLAMA_EXPERTFLOW_SPLIT_PROFILE", None)
    result.pop("GGML_SCHED_DEBUG", None)
    result["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    if mode == "on":
        result["LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER"] = STATIC_LAYERS
    return result


def runtime_environment(
    inherited: dict[str, str], mode: str, executable: Path
) -> dict[str, str]:
    result = build_environment(inherited, mode)
    cuda_root = Path(result.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"))
    prefixes = [str(executable.parent.resolve()), str((cuda_root / "bin").resolve())]
    result["PATH"] = os.pathsep.join(prefixes + [result.get("PATH", "")])
    return result


def build_server_command(executable: Path, model: Path, port: int, log_file: Path) -> list[str]:
    return [
        str(executable), "-m", str(model), "-ngl", "99", "--cpu-moe",
        "-c", "2048", "-b", "2048", "-ub", "512", "-t", "12", "-tb", "12",
        "-np", "1", "--host", "127.0.0.1", "--port", str(port),
        "--metrics", "--log-file", str(log_file),
    ]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stream_metrics(request_start: float, events: list[tuple[float, dict[str, object]]]) -> dict[str, object]:
    arrivals: list[float] = []
    token_ids: list[int] = []
    content: list[str] = []
    timings: dict[str, object] = {}
    for arrived, event in events:
        event_tokens = event.get("tokens")
        if isinstance(event_tokens, list) and event_tokens:
            arrivals.append(arrived)
            token_ids.extend(int(token) for token in event_tokens)
        if isinstance(event.get("content"), str):
            content.append(str(event["content"]))
        if isinstance(event.get("timings"), dict):
            timings = dict(event["timings"])
    latencies = [round((right - left) * 1000.0, 6) for left, right in zip(arrivals, arrivals[1:])]
    ttft = round((arrivals[0] - request_start) * 1000.0, 6) if arrivals else None
    text = "".join(content)
    return {
        "ttft_ms": ttft,
        "token_ids": token_ids,
        "token_latency_ms": latencies,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "prompt_tps": timings.get("prompt_per_second"),
        "decode_tps": timings.get("predicted_per_second"),
        "prompt_tokens": timings.get("prompt_n"),
        "generated_tokens": timings.get("predicted_n"),
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode("utf-8")).hexdigest(),
    }


def gpu_state() -> dict[str, int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    values = [int(value.strip()) for value in completed.stdout.strip().split(",")]
    return dict(zip(("temperature_c", "sm_clock_mhz", "used_mib", "total_mib"), values, strict=True))


def wait_ready(port: int, process: subprocess.Popen[bytes], timeout_seconds: float = 40.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before health check: {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("server health check timed out")


def request_stream(port: int, prompt: str, n_predict: int) -> tuple[float, float, list[tuple[float, dict[str, object]]]]:
    payload = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 42,
        "ignore_eos": True, "cache_prompt": False, "stream": True,
        "return_tokens": True, "timings_per_token": True,
    })
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
    started = time.perf_counter()
    connection.request("POST", "/completion", body=payload, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"completion HTTP {response.status}: {response.read().decode('utf-8', errors='replace')}")
    events: list[tuple[float, dict[str, object]]] = []
    while True:
        line = response.readline()
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        event = json.loads(line[6:].decode("utf-8"))
        events.append((time.perf_counter(), event))
        if event.get("stop") is True:
            break
    finished = time.perf_counter()
    connection.close()
    return started, finished, events


def run_one(
    executable: Path, model: Path, output_dir: Path, sampler: Path,
    mode: str, pair_index: int, order_index: int, port: int, prompt: str, n_predict: int,
) -> dict[str, object]:
    run_dir = output_dir / f"pair-{pair_index + 1:02d}-{order_index + 1}-{mode}"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = build_server_command(executable, model, port, run_dir / "runtime.log")
    environment = runtime_environment(dict(os.environ), mode, executable)
    before = gpu_state()
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    monitor_path = run_dir / "gpu-samples.json"
    process: subprocess.Popen[bytes] | None = None
    monitor: subprocess.Popen[bytes] | None = None
    started_wall = time.perf_counter()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       env=environment, creationflags=creation_flags)
            monitor = subprocess.Popen([
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(sampler),
                "-ProcessId", str(process.pid), "-Output", str(monitor_path), "-IntervalMs", "200",
            ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=creation_flags)
            wait_ready(port, process)
            request_start, request_end, events = request_stream(port, prompt, n_predict)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if monitor is not None:
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=5)
    elapsed = time.perf_counter() - started_wall
    if not monitor_path.is_file():
        raise RuntimeError("GPU monitor did not produce evidence")
    metrics = stream_metrics(request_start, events)
    result: dict[str, object] = {
        "schema_version": "1.0.0", "pair": pair_index + 1, "order": order_index + 1, "mode": mode,
        "command": command, "environment": {"LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER": environment.get("LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER")},
        "gpu_before": before, "gpu_monitor": json.loads(monitor_path.read_text(encoding="utf-8-sig")),
        "request_seconds": request_end - request_start, "cold_process_seconds": elapsed,
        "return_code_after_termination": process.returncode if process is not None else None,
        **metrics,
    }
    (run_dir / "measurement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched cold-process Q6 OFF/ON pairs")
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--port-base", type=int, default=18100)
    parser.add_argument("--prompt", default="Caching.")
    parser.add_argument("--n-predict", type=int, default=512)
    args = parser.parse_args()
    if not args.server.is_file() or not args.model.is_file():
        raise FileNotFoundError("server or model is missing")
    if file_sha256(args.model) != MODEL_SHA256:
        raise ValueError("Q6 model SHA-256 mismatch")
    sampler = Path(__file__).with_name("sample_q6_gpu.ps1").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    run_number = 0
    for pair_index, modes in enumerate(pair_order(args.pairs)):
        for order_index, mode in enumerate(modes):
            run_number += 1
            results.append(run_one(
                args.server.resolve(), args.model.resolve(), args.output_dir.resolve(), sampler,
                mode, pair_index, order_index, args.port_base + run_number, args.prompt, args.n_predict,
            ))
    (args.output_dir / "raw-results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
