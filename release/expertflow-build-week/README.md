# ExpertFlow

Profile-guided CUDA placement for quantized Mixture-of-Experts models.

**PRODUCT RELEASE:** verified Gemma 4 26B A4B Q6_K profile for Windows and NVIDIA CUDA.

ExpertFlow's Build Week product release targets one verified case: Gemma 4 26B A4B Q6_K on a 16 GB NVIDIA GPU. It profiles routed-expert cost, selects complete expert banks that fit the VRAM budget, and writes a reproducible deployment manifest. The runtime feature is off unless a deployment enables it.

## Installation

Requirements for the model-free replay:

- Windows, Linux, or macOS
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

Clone the repository and create the pinned environment:

```powershell
git clone https://github.com/crankysmh47/Expertflow.git
cd Expertflow
uv sync --extra dev
```

Live Q6 inference additionally needs an NVIDIA CUDA system, the compatible ExpertFlow llama.cpp build, and a user-supplied `google_gemma-4-26B-A4B-it-Q6_K.gguf`. The model is not distributed with this repository.

## Quick start

Replay the committed result without a GPU or GGUF:

```powershell
uv run expertflow demo --replay
```

This verifies the evidence hashes and prints the stock/ExpertFlow comparison, quality status, layer plan, VRAM result, and caching decision. It is evidence replay, not a live benchmark.

Open the offline product report:

```powershell
Start-Process docs/evidence/product-release/dashboard.html
```

## What ExpertFlow does

Gemma's routed expert tensors are large enough to dominate CPU time, but whole-layer GPU offload is an awkward fit for a 16 GB card. ExpertFlow profiles those expert banks separately from the rest of each transformer block. For the winning Q6 deployment, it keeps the normal CPU sources and creates full static CUDA identity shadows for layers `0–9`, `15`, and `20`.

There is no runtime cache in this release. Every selected layer has all 128 experts resident, so there is no eviction, prediction, or per-token H2D transfer. A bounded predictive-cache study found no worthwhile cache operating point on this hardware.

## Verified result

The main claim comes from ten matched 512-token runs:

| Result | Stock | ExpertFlow |
|---|---:|---:|
| Decode TPS | 22.967 | 28.13 |
| Relative change | — | +22.48% |
| Peak process-owned VRAM | 3,098.656 MiB | 10,966.801 MiB |
| Static expert layers | none | 12 |

Quality is mixed but fully disclosed. The perplexity point estimate improved by 2.92%, and MMLU moved from 49/100 to 50/100. The 95% perplexity upper bound was +2.25%, so the conservative +1% confidence requirement did not pass.

The product stage added two separate live server profiles:

| Profile | Measured result | Caveat |
|---|---:|---|
| Four-slot throughput | 35.67 aggregate generated TPS | Five cold-server repetitions; concurrent outputs were not deterministic across all runs |
| Context | 262,144 allocated tokens | 417 tokens were processed; the full allocation was not filled |

Do not compare 35.67 aggregate server TPS with 28.13 single-request CLI TPS as though they were the same protocol.

## CLI

### `expertflow doctor`

Inspect Windows, NVIDIA GPU/VRAM, driver, CUDA runtime, model, and binary hashes:

```powershell
uv run expertflow doctor `
  --model $env:EXPERTFLOW_MODEL_PATH `
  --runtime $env:EXPERTFLOW_LLAMA_CLI `
  --server $env:EXPERTFLOW_LLAMA_SERVER
```

### `expertflow profile <model>`

Show the committed measured Q6 profile and a live model-path/hash check:

```powershell
uv run expertflow profile $env:EXPERTFLOW_MODEL_PATH
```

### `expertflow optimize <model> --goal ...`

Write one of the measured deployment profiles:

```powershell
uv run expertflow optimize $env:EXPERTFLOW_MODEL_PATH --goal latency --output deployment.json
uv run expertflow optimize $env:EXPERTFLOW_MODEL_PATH --goal throughput --output deployment.json
uv run expertflow optimize $env:EXPERTFLOW_MODEL_PATH --goal context --output deployment.json
uv run expertflow optimize $env:EXPERTFLOW_MODEL_PATH --goal agentic --output deployment.json
```

### `expertflow run <deployment.json>`

Launch the reproducible single-request command:

```powershell
uv run expertflow run deployments/max-performance.json
```

Use `--dry-run` to inspect the command without loading the model.

### `expertflow serve <deployment.json>`

Launch llama-server and print the base URL, health URL, context, slots, placement, and expected peak VRAM:

```powershell
uv run expertflow serve deployments/max-agentic.json
```

### `expertflow compare <deployment.json>`

Print the recorded stock comparison and quality status:

```powershell
uv run expertflow compare deployments/max-performance.json
```

### `expertflow demo --replay`

Verify and replay the committed evidence on an ordinary machine:

```powershell
uv run expertflow demo --replay
```

## Environment

Copy `.env.example` into your shell configuration or set these variables directly:

| Variable | Purpose |
|---|---|
| `EXPERTFLOW_MODEL_PATH` | User-supplied Q6_K GGUF |
| `EXPERTFLOW_LLAMA_CLI` | Compatible `llama-cli.exe` |
| `EXPERTFLOW_LLAMA_SERVER` | Compatible `llama-server.exe` |
| `EXPERTFLOW_BASE_URL` | OpenAI-compatible base URL; default is `http://127.0.0.1:8080/v1` |

Verify the model before live inference:

```powershell
Get-FileHash $env:EXPERTFLOW_MODEL_PATH -Algorithm SHA256
```

Expected SHA-256:

```text
089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba
```

## Agentic example

Start the measured four-slot profile:

```powershell
./scripts/start_expertflow.ps1 -Deployment deployments/max-agentic.json
Invoke-RestMethod http://127.0.0.1:8080/health
uv run python examples/openai_client.py
uv run python examples/agentic_session.py
./scripts/stop_expertflow.ps1
```

The examples work with clients that accept a configurable OpenAI-compatible base URL. They do not claim compatibility with tools that force a specific provider.

## Reproducible build

The release package contains the pinned upstream SHA, ExpertFlow patch series, build flags, compiler/CUDA versions, binary hashes, and setup scripts. Start with `submission/judge-test-guide.md` for the three judge paths.

## Tests

```powershell
uv sync --all-extras
$env:PYTHONPATH = "$PWD;$PWD\src"
uv run python -m pytest -q `
  --ignore=tests/test_t1_temporal_source_contract.py `
  --ignore=tests/test_t2_sidecar_source_contract.py
```

Historical T1/T2 temporal source-contract tests target a different preserved llama.cpp branch. The release verification records that exclusion instead of pointing those tests at the static Q6 branch.

## License

MIT. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
