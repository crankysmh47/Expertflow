# ExpertFlow Q4 Observatory Design

**Status:** Approved for implementation on 2026-07-14  
**Track:** Developer Tools  
**Primary machine:** Windows 11, RTX 5060 Ti 16 GiB, 31.1 GiB system RAM  
**Primary model profile:** Gemma 4 26B A4B Q4 GGUF, text-only, batch size 1

## Product decision

ExpertFlow is an Observatory first and a live predictive runtime only after evidence supports it. It profiles real sparse-MoE routing, models cache and transfer deadlines, recommends a machine-specific expert-residency budget, and explains ready, late, and blocking expert events through a causal replay.

The hackathon submission floor is a complete runnable Observatory. Live expert movement is a competitive extension, not a dependency of the product story.

## Locked architecture

### Real-model path

- A pinned Gemma 4 26B A4B Q4 GGUF is the canonical deployment artifact.
- Official pinned llama.cpp CUDA release binaries perform the unmodified GPU baseline; an exact source archive at the corresponding pinned revision supplies the inspection and instrumentation base.
- The first C/C++ change may only export routing telemetry. It must not mutate cache placement, allocation, routing decisions, or generated tokens.
- The initial trace contract records token index, layer index, selected expert identifiers, optional routing weights, and a monotonic timestamp.
- Instrumented and uninstrumented deterministic runs must generate identical token sequences.

### Product path

- Python validates trace records and manifests.
- Python computes locality, concentration, reuse distance, layer working sets, and policy simulations.
- Python emits machine-specific recommendations and a causal replay report.
- A tiny Gemma 4 checkpoint supplies fast schema and CI tests; it never substitutes for the real Q4 evidence.

### Artifact boundary

Model weights and generated run data remain outside Git. Git contains only pinned manifests, checksums, scripts, test fixtures small enough for review, expected-result manifests, source code, and documentation.

## Twenty-four-hour feasibility gate

The llama.cpp path proceeds only if all five checks pass:

1. The pinned Q4 GGUF loads and produces text on the target machine.
2. The Gemma 4 MoE routing operation can be identified in the pinned llama.cpp/GGML source.
3. A minimal telemetry-only patch can export selected experts for each token and MoE layer.
4. Instrumented and uninstrumented deterministic outputs match exactly.
5. Real traces show enough locality for at least one bounded cache policy to improve a relevant movement or deadline metric over reactive/static behavior.

Failure to pass checks 1–4 within 24 hours freezes the live llama.cpp fork. The fallback remains a complete Observatory using the strongest real routing evidence obtainable without a dangerous multi-day runtime fork.

## Memory accounting contract

Every real run records:

- GGUF file size and SHA-256;
- process committed memory and working set;
- GPU memory before load, after load, and at peak generation;
- configured GPU layer count;
- context length, batch size, and generated token count;
- KV/state allocation when exposed by the runtime;
- temporary compute buffers when exposed by the runtime;
- trace-writer memory and output size;
- operating-system and background GPU use at run start.

The public report may say the model fits only after these measurements exist. File size alone is not a runtime-memory claim.

## Exactness and failure behavior

- The true router remains authoritative.
- Telemetry failure must fail visibly or disable tracing explicitly; it must never silently change routing.
- Malformed or incomplete trace records are rejected with a record number and reason.
- Hash, size, model revision, llama.cpp revision, command line, and environment details are mandatory in measured run manifests.
- Simulated metrics are labeled `estimated`; runtime metrics are labeled `measured`.

## Clean repository structure

```text
configs/                 pinned model and runtime manifests
docs/superpowers/        approved designs and execution plans
docs/evidence/           compact source maps and gate decisions
scripts/                 reproducible setup, download, and benchmark entry points
src/expertflow/          Python product package
tests/                   unit and integration tests
PROJECT_LOG.md           append-oriented execution history
```

Generated weights, third-party source trees, runtime binaries, builds, runs, and reports are excluded from Git and stored beneath `C:\models\expertflow`. Their pinned identities and checksums remain in `configs/`.

## Scope exclusions

- Q8 is not on the hackathon critical path.
- No cache mutation is attempted before routing telemetry and parity pass.
- No neural predictor, MTP controller, multimodal input, KV-cache optimization, or multi-user scheduler is part of the initial gate.
- No speedup, exactness, or memory-fit claim is public until supported by recorded artifacts.
