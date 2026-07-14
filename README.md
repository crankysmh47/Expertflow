# ExpertFlow Local

A hardware-aware routing observatory for running sparse mixture-of-experts models on one local GPU.

ExpertFlow is an OpenAI Build Week project in active development. The small analysis path works today; the real Gemma 4 Q4 baseline is still being collected. See [current status](#current-status) before treating any result as final.

## Installation

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). The analysis commands are CPU-only. A compatible NVIDIA GPU and the pinned llama.cpp runtime are needed only for the real baseline.

```powershell
git clone https://github.com/crankysmh47/Expertflow.git
cd Expertflow
uv sync --extra dev
uv run pytest
```

Model weights, runtime binaries, source archives, and generated runs stay outside Git. On the development machine they live under `C:\models\expertflow`.

## Quick start

Check the machine and write a JSON report:

```powershell
uv run expertflow doctor `
  --artifact-root C:\models\expertflow `
  --output C:\models\expertflow\runs\preflight\doctor.json
```

You should get GPU memory, system RAM, disk space, Python/platform details, and resolved tool paths. Missing tools such as `nvcc` appear as `null`; the command does not quietly change the runtime plan.

## What is ExpertFlow?

Sparse MoE models select a small set of experts for each token and layer. That saves compute, but the full expert set can still be awkward to place on a 16 GB GPU. ExpertFlow records the router's actual selections, measures locality, compares cache policies under a fixed slot budget, and produces machine-specific evidence for a residency decision.

The project starts as an observatory. Live expert movement is gated behind real routing traces, deterministic output parity, and a measured benefit over reactive and LRU baselines.

## Why this approach?

A model file fitting on disk says little about runtime memory. Context state, compute buffers, CUDA libraries, and desktop GPU use all matter. ExpertFlow records those costs instead of inferring them from GGUF size.

The runtime boundary is deliberately small. In pinned llama.cpp source, Gemma 4 names the selected-expert tensor `ffn_moe_topk-{layer}`. The existing evaluation callback can copy that tensor without changing router decisions or graph semantics. If parity or locality fails, the project remains a useful trace profiler and simulator rather than forcing a risky runtime fork.

## Current status

| Area | Status |
| --- | --- |
| Hardware doctor | Working; writes measured JSON |
| Strict router-event schema | Working; malformed records fail with record numbers |
| Locality profile | Working; measured concentration and reuse metrics |
| Policy simulation | Working; reactive, static-hotset, and LRU results labeled estimated |
| Pinned llama.cpp CUDA runtime | Verified; release `b10002`, CUDA 12.4, RTX 5060 Ti visible |
| Gemma 4 Q4 artifact | Verified; 14,439,361,440 bytes and pinned SHA-256 |
| Unmodified real-model baseline | Passed on CPU and bounded 10-layer GPU offload |
| Token parity comparison | Working; exact measured comparison with first mismatch |
| Routing callback probe | Passed; 1,350 real events and exact token parity |
| GPT-5.6 explanation layer | Pending; no API-backed claim yet |

The append-only [project log](PROJECT_LOG.md) records commands, failures, decisions, hashes, and test results as the work moves.

## CLI

```text
expertflow baseline  Run and measure an unmodified llama.cpp baseline
expertflow doctor    Record hardware, storage, and toolchain readiness
expertflow profile   Build a measured locality profile from router JSONL
expertflow parity    Compare exact token sequences with and without tracing
expertflow recommend Produce an evidence-bounded machine recommendation
expertflow simulate  Compare estimated cache policies under one slot budget
```

Run `uv run expertflow <command> --help` for the full option list.

### Fetch the pinned Q4 model

The fetcher uses the exact Hugging Face revision in `configs/model-artifacts.toml`, resumes partial transfers, and verifies byte length plus SHA-256 before returning success.

```powershell
uv run expertflow-fetch-q4 --destination C:\models\expertflow
```

Expected verified path:

```text
C:\models\expertflow\google--gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf
```

### Run the measured baseline

The official runtime artifacts and hashes are pinned in `configs/runtime-artifacts.toml`. Once those files and the Q4 model are present, run:

```powershell
uv run expertflow baseline `
  --runtime C:\models\expertflow\dependencies\llama-b10002\runtime\llama-cli.exe `
  --model C:\models\expertflow\google--gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf `
  --model-sha256 4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5 `
  --prompt-file configs\baseline-prompt.txt `
  --output-dir C:\models\expertflow\runs\q4-auto `
  --gpu-layers auto --ctx-size 1024 --predict 32 --threads 12
```

The run directory contains `manifest.json`, `stdout.txt`, `stderr.txt`, and `llama.log`. The manifest records the exact command, model identity, elapsed time, process memory, and sampled GPU memory.

### Build the routing probe

The separate probe links to the same verified b10002 CUDA DLLs as the baseline. Its build tree stays outside Git:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_router_probe.ps1 `
  -LlamaCppSource C:\models\expertflow\dependencies\llama.cpp-a7312ae94f801fc9c6786dc56e38df57b964f697
```

Build provenance and the callback boundary are recorded in [router probe evidence](docs/evidence/router-probe-build.md).

The first real Q4 measurements, token parity, and conditional live-cache verdict are in [Q4 baseline evidence](docs/evidence/q4-baseline-result.md).

### Profile a router trace

```powershell
uv run expertflow profile C:\models\expertflow\runs\trace.jsonl `
  --static-budget 1 --static-budget 2 --static-budget 4 `
  --output C:\models\expertflow\runs\profile.json
```

Profile output is labeled `measured` because it summarizes observed router events. It includes expert concentration, static hit curves, adjacent-token reuse, and mean reuse distance by layer.

Compare the deterministic probe's tracing-disabled and tracing-enabled token artifacts with:

```powershell
uv run expertflow parity baseline-tokens.json instrumented-tokens.json `
  --output C:\models\expertflow\runs\parity.json
```

### Compare cache policies

```powershell
uv run expertflow simulate C:\models\expertflow\runs\trace.jsonl `
  --capacity-per-layer 4 `
  --output C:\models\expertflow\runs\simulation.json
```

Simulation output is labeled `estimated`. The current policies are reactive, trace-derived static hotset, and online per-layer LRU. Slot capacity must be at least the router top-k.

### Produce a machine-specific recommendation

```powershell
uv run expertflow recommend `
  --doctor C:\models\expertflow\runs\preflight\doctor.json `
  --baseline C:\models\expertflow\runs\q4-gpu10-smoke8\manifest.json `
  --profile C:\models\expertflow\runs\q4-probe\profile.json `
  --simulation C:\models\expertflow\runs\q4-probe\simulation.json `
  --output C:\models\expertflow\runs\q4-probe\recommendation.json
```

The current real recommendation is `CONDITIONAL`: use static-hotset for replay, but keep live caching disabled until expert byte size, transfer timing, and a stratified trace are available.

## Trace contract

Router traces use strict JSONL. One record represents one token/layer decision from the authoritative router:

```json
{
  "schema_version": "1.0.0",
  "request_id": "req-001",
  "conversation_id": "conv-001",
  "turn_index": 0,
  "phase": "decode",
  "forward_id": 12,
  "hook_order": 41,
  "token_index": 20,
  "token_id": 42,
  "layer_id": 7,
  "selected_expert_ids": [2, 9],
  "selected_expert_weights": [0.75, 0.25],
  "observed_at_ns": 123456789
}
```

Unknown fields, unsupported schema versions, duplicate experts, invalid ranges, and mismatched weight arrays fail visibly. Raw prompt text is not part of the routing record.

## Architecture

```text
Gemma 4 router callback
        |
        v
strict JSONL trace ----> measured locality profile
        |                         |
        v                         v
reactive/static/LRU simulator -> machine recommendation (next stage)
        |
        v
causal replay and GPT-5.6 explanation (next stage)
```

The repository keeps reviewable code and manifests. Large or generated material stays under the external artifact root.

```text
configs/             pinned model/runtime identities and baseline prompt
docs/evidence/       compact runtime and routing-source evidence
docs/superpowers/    approved design and execution plan
src/expertflow/      CLI, validation, analysis, simulation, measurement
tests/               CPU-fast unit and CLI integration tests
PROJECT_LOG.md       chronological execution record
```

## OpenAI Build Week integration

Codex is being used to evaluate the plan, inspect the machine, implement the project, run tests, and maintain `PROJECT_LOG.md`. The final submission will include the required Codex session identifier from `/feedback`.

GPT-5.6 will receive validated doctor/profile/simulation artifacts and produce a grounded explanation and recommended configuration. That layer is intentionally downstream of deterministic measurements. It is marked pending above because the repository does not yet contain the API integration.

## Further reading

- [Approved Q4 Observatory design](docs/superpowers/specs/2026-07-14-expertflow-q4-observatory-design.md)
- [Stage 0 execution plan](docs/superpowers/plans/2026-07-14-stage0-q4-baseline.md)
- [Pinned llama.cpp baseline evidence](docs/evidence/llama-baseline.md)
- [Gemma 4 routing source map](docs/evidence/gemma4-routing-source-map.md)

## License

This hackathon prototype does not have a repository license yet. Do not assume redistribution rights for the code, model weights, or third-party runtime artifacts.

## Development

Run the complete test suite before committing:

```powershell
uv run pytest -q
```

Generated caches, builds, model files, runtime binaries, and run artifacts are ignored by Git. Keep measured evidence outside the repository and commit only compact manifests or reports needed for review.
