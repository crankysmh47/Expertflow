# ExpertFlow deployment and local hardware guide

Source: <https://github.com/crankysmh47/Expertflow>

The repository contains two independent surfaces:

1. A static, model-free dashboard and evidence replay that work on ordinary developer machines.
2. The verified live CUDA path for compatible NVIDIA hardware.

The GGUF is not included. The model is needed only for live inference and benchmarking.

## Deploy the dashboard to Vercel

The dashboard is dependency-free HTML, CSS, and JavaScript. `vercel.json` routes both `/` and `/dashboard` to the self-contained release dashboard.

### GitHub import

1. Open <https://vercel.com/new>.
2. Import `crankysmh47/Expertflow`.
3. Leave the repository root as `.`.
4. Choose **Other** for the framework preset.
5. Leave Build Command and Output Directory empty.
6. Deploy.

### Vercel CLI

From the repository root:

```powershell
npx vercel --prod
```

No model, CUDA installation, secret, or environment variable is required for the hosted dashboard.

## Run the dashboard locally

```powershell
git clone https://github.com/crankysmh47/Expertflow.git
cd Expertflow
py -m http.server 8767 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8767/docs/evidence/product-release/dashboard.html
```

## Replay the verified evidence on any computer

Requirements: Python 3.11 or newer, Git, and `uv`.

```powershell
git clone https://github.com/crankysmh47/Expertflow.git
cd Expertflow
uv sync --frozen
uv run expertflow demo --replay
```

Expected output includes `status=pass`, `evidence_hashes_verified=true`, `22.967` stock TPS, `28.13` ExpertFlow TPS, and `22.48%` improvement. This is a hash-verified recorded-evidence replay, not live inference.

## Run on compatible local NVIDIA hardware

The verified configuration is Windows 11 x64, NVIDIA RTX 5060 Ti 16 GB, driver 591.86, CUDA 12.8.93, and Gemma 4 26B A4B IT Q6_K. Other Windows/NVIDIA systems are compatible but unverified. Linux x64 + NVIDIA is experimental. macOS, AMD, and CPU-only systems support the evidence replay but not the shipped CUDA acceleration path.

### 1. Obtain the model

Use `google_gemma-4-26B-A4B-it-Q6_K.gguf`. Verify:

```text
SHA-256: 089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba
```

### 2. Obtain or build the pinned runtime

The ordered llama.cpp patch series, upstream pin, compiler/CUDA metadata, and build script are included in the repository. See [`JUDGES.md`](JUDGES.md) and [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

```powershell
.\scripts\build_release_runtime.ps1 -Source C:\path\to\llama.cpp -Build C:\path\to\build
```

### 3. Configure paths

```powershell
$env:EXPERTFLOW_MODEL_PATH = "C:\path\to\google_gemma-4-26B-A4B-it-Q6_K.gguf"
$env:EXPERTFLOW_LLAMA_CLI = "C:\path\to\llama-cli.exe"
$env:EXPERTFLOW_LLAMA_SERVER = "C:\path\to\llama-server.exe"
```

### 4. Inspect compatibility

```powershell
uv run expertflow doctor `
  --model $env:EXPERTFLOW_MODEL_PATH `
  --runtime $env:EXPERTFLOW_LLAMA_CLI `
  --server $env:EXPERTFLOW_LLAMA_SERVER
```

### 5. Run the live matched TPS demonstration

```powershell
.\scripts\live-tps-demo.ps1 -Mode Demo
```

The command verifies the expected model and runtime hashes, launches fresh matched stock and ExpertFlow processes, prints the results, and saves the raw evidence. Use `-Mode Judge` for three matched pairs.

### 6. Generate and run a deployment

```powershell
uv run expertflow profile $env:EXPERTFLOW_MODEL_PATH
uv run expertflow optimize $env:EXPERTFLOW_MODEL_PATH --goal max-performance --output deployment.json
uv run expertflow run deployment.json --model $env:EXPERTFLOW_MODEL_PATH
uv run expertflow compare deployment.json
```

### 7. Start the OpenAI-compatible server

```powershell
uv run expertflow optimize $env:EXPERTFLOW_MODEL_PATH --goal agentic --output deployment.json
uv run expertflow serve deployment.json
uv run python examples/agentic_session.py
```

## Platform matrix

| Platform | Dashboard/replay | Live ExpertFlow CUDA |
|---|---:|---:|
| Windows 11 x64 + RTX 5060 Ti 16 GB | Supported | Verified |
| Other Windows x64 + NVIDIA CUDA | Supported | Compatible, unverified |
| Linux x64 + NVIDIA | Supported | Experimental |
| macOS / Metal | Supported | Replay only |
| AMD Vulkan or ROCm | Supported | Replay only |
| CPU-only | Supported | Replay only |

For exact benchmark comparability, use the pinned model, runtime, seed, prompt, batch settings, CUDA graph setting, and repetition protocol described in [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).
