# Judge test guide

The GGUF is not included. Live paths require the named Q6_K model and a compatible Windows/NVIDIA system.

## Level 1: evidence replay

No model or NVIDIA GPU is needed.

```powershell
uv sync --extra dev
uv run expertflow demo --replay
Start-Process docs/evidence/product-release/dashboard.html
```

Expected: `status=pass`, `evidence_hashes_verified=true`, 22.967 stock TPS, 28.13 ExpertFlow TPS, and the disclosed quality/cache status.

## Level 2: prebuilt compatible system

```powershell
$env:EXPERTFLOW_MODEL_PATH = "C:\path\to\google_gemma-4-26B-A4B-it-Q6_K.gguf"
$env:EXPERTFLOW_LLAMA_CLI = "C:\path\to\llama-cli.exe"
$env:EXPERTFLOW_LLAMA_SERVER = "C:\path\to\llama-server.exe"

uv run expertflow doctor --model $env:EXPERTFLOW_MODEL_PATH --runtime $env:EXPERTFLOW_LLAMA_CLI --server $env:EXPERTFLOW_LLAMA_SERVER
./scripts/live-tps-demo.ps1 -Mode Demo
uv run expertflow run deployments/max-performance.json
uv run expertflow compare deployments/max-performance.json
```

The live TPS command runs one fresh matched pair and writes its evidence directory. It is a quick judge-visible check, not a replacement for the committed ten-pair result.

Model SHA-256 must equal `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.

## Level 3: reproducible build

Use the patch series in `patches/llama.cpp`, upstream `a7312ae94f801fc9c6786dc56e38df57b964f697`, MSVC v143 14.39.33519, CUDA 12.8.93, CMake, and Ninja. Run `scripts/build_release_runtime.ps1`, verify the binary hashes, then repeat Level 2.

## Other CLI paths

```powershell
uv run expertflow profile $env:EXPERTFLOW_MODEL_PATH
uv run expertflow optimize $env:EXPERTFLOW_MODEL_PATH --goal latency --output deployment.json
uv run expertflow serve deployments/max-agentic.json --dry-run
```

If a live result differs, preserve the logs and compare model, binary, driver, context, batch, threads, CUDA graph setting, and environment variables before interpreting it.
