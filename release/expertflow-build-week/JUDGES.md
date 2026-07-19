# Judge ExpertFlow

All headline values come from `release/expertflow-build-week/evidence/release-scorecard.json`.

## Path A: any computer, under two minutes

Requires Python 3.11+ and `uv`. It needs no model, NVIDIA GPU, CUDA toolkit, or compiler.

```console
uv sync --frozen
uv run expertflow demo --replay
```

Expected output verifies the evidence hash and reports 22.967 stock TPS, 28.13 ExpertFlow TPS, 22.48% improvement, layers `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20]`, 10,966.801 MiB peak process-owned VRAM, quality status, and the simulated `NO CACHE OPPORTUNITY` decision. Open `release/expertflow-build-week/dashboard.html` for the offline report.

Trust notes: the strict PPL confidence gate was not met; MMLU moved from 49/100 to 50/100; four-slot outputs were not fully deterministic; and the 262,144-token context allocation processed 417 tokens rather than a full context.

## Path B: compatible live system

The verified live platform is Windows 11 x64, NVIDIA RTX 5060 Ti 16 GB, driver 591.86, and CUDA 12.8.93. Supply the Q6_K GGUF yourself and set the runtime paths.

```powershell
$env:EXPERTFLOW_MODEL_PATH = "C:\path\to\google_gemma-4-26B-A4B-it-Q6_K.gguf"
$env:EXPERTFLOW_LLAMA_CLI = "C:\path\to\llama-cli.exe"
$env:EXPERTFLOW_LLAMA_SERVER = "C:\path\to\llama-server.exe"
uv run expertflow doctor
uv run expertflow run deployments/max-performance.json --model $env:EXPERTFLOW_MODEL_PATH
uv run expertflow compare deployments/max-performance.json
```

The expected model SHA-256 is `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.

## Path C: full reproduction

Allow 30–60 minutes for the CUDA build and roughly 15 minutes for the ten 512-token matched pairs on comparable hardware. Different hardware may produce different TPS.

1. Check out llama.cpp `a7312ae94f801fc9c6786dc56e38df57b964f697`.
2. Install VS 2022 v143 14.39.33519, CUDA 12.8.93, CMake, and Ninja.
3. Run `scripts/build_release_runtime.ps1 -Source <llama-source> -Build <build-dir>`.
4. Verify CLI SHA-256 `5d68046dcd26e2fd018aaeaad5f99cdb7d88eca6fc10935925f1d660f7009407` and server SHA-256 `22ecc4f64f91dcbe3a1cfe7d9d4617e43467ea7f3c6fa1ba2c6ad8d07e89334e`.
5. Verify the model hash, then follow `docs/BENCHMARKING.md` and `docs/evidence/q6-placement-final/reproduction-commands.md`.

The patch series ends at llama.cpp `451224ab4d12a616dc3e16e8c8063f4b331f531c`. Live acceleration outside the verified Windows/NVIDIA setup is not claimed.
