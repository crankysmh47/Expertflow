# Stratified Gemma 4 Q4 routing evidence

Date: 2026-07-15 PKT

This checkpoint broadens the original one-prompt feasibility trace with five public synthetic prompt shapes. Every case used the verified Gemma 4 26B A4B Q4 artifact, the same b10002 probe, 10 GPU layers, 12 CPU threads, greedy sampling, and eight generated tokens. Tracing-disabled and tracing-enabled runs were executed as separate processes.

## Prompt set

| Slug | Shape |
| --- | --- |
| `code-python-lru` | Code generation plus a short invariant explanation |
| `short-factual-tides` | Short factual response with a length constraint |
| `constrained-gpu-plan` | Multi-constraint technical planning |
| `incident-summary` | Summarization into three bullets |
| `bilingual-cache` | Urdu explanation plus an English summary |

The trace records token IDs and router selections, not raw prompt text. Generated artifacts remain outside Git under `C:\models\expertflow\runs\stratified-q4`.

## Results

| Prompt | Exact generated-token parity | Events | Expert demands | Mean adjacent reuse | Static-8 hit rate | LRU-8 hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `code-python-lru` | PASS | 810 | 6,480 | 35.38% | 38.87% | 34.07% |
| `short-factual-tides` | FAIL at token 2 | 600 | 4,800 | 34.52% | 39.83% | 32.79% |
| `constrained-gpu-plan` | FAIL at token 2 | 1,410 | 11,280 | 36.56% | 36.37% | 35.78% |
| `incident-summary` | PASS | 1,260 | 10,080 | 31.13% | 39.75% | 30.39% |
| `bilingual-cache` | FAIL at token 1 | 810 | 6,480 | 33.22% | 40.77% | 31.99% |
| Demand-weighted aggregate | 2/5 PASS | 4,890 | 39,120 | — | 38.81% | 33.11% |

The policy numbers are replay estimates, not measured latency or transfer savings.

## Parity diagnosis

For each failed case, a second tracing-disabled run exactly matched the first tracing-disabled run, and a second tracing-enabled run exactly matched the first tracing-enabled run. The repeated tracing-disabled and tracing-enabled outputs still diverged at the same early token. Prompt token IDs always matched.

A CPU-only control pair for `short-factual-tides` passed exact prompt and generated-token parity across 600 events. This isolates the observed divergence to the GPU tensor-observation graph path rather than sampling randomness, prompt reconstruction, trace serialization, or the token comparator. The current evidence does not yet establish the lower-level CUDA scheduling mechanism.

## CUDA callback gate

The router trace remains structurally valid and repeatable, but the GPU observation path does not satisfy exact output transparency across the stratified workload. The earlier one-prompt telemetry PASS is therefore insufficient as a general gate.

The CUDA callback path cannot be used as transparent telemetry for this workload. llama.cpp's scheduler executes a whole backend split when no evaluation callback is installed. With the callback installed, it instead computes synchronized graph views ending at every requested router tensor. The observed parity result is consistent with that changed GPU execution path; the exact lower-level floating-point mechanism is not claimed.

## Pinned Vulkan telemetry control

The official b10002 Windows Vulkan archive was downloaded from the same release and verified before extraction:

- file: `llama-b10002-bin-win-vulkan-x64.zip`
- bytes: `32,950,223`
- SHA-256: `c2b66ab6912e9fad75c7c6d2000f660bb40bc1f063aa62ec671d471e27dd92ea`
- GPU selected by the backend: NVIDIA GeForce RTX 5060 Ti

The unchanged probe binary was placed beside the ABI-compatible b10002 Vulkan DLLs. All five paired runs passed exact prompt and generated-token parity, producing the same 4,890-event workload shape.

For the fixed prompt-prefill portion, CUDA and Vulkan selected exactly the same ordered eight-expert set in 84.74% of 3,840 token/layer events. Individual expert selections overlapped 99.26% across 30,720 demands. The demand-weighted prefill policy estimates are nearly identical:

| Backend | Static-8 | LRU-8 |
| --- | ---: | ---: |
| CUDA observation path | 40.6868% | 33.1283% |
| Vulkan observation path | 40.6803% | 33.1641% |

The static difference is 0.0065 percentage points and the LRU difference is 0.0358 points. Generated-continuation events are excluded from this cross-backend comparison because the failed CUDA callback cases produced different continuations. These are demand-weighted prompt-local simulations: each prompt receives its own fitted static hotset and a reset LRU cache.

A later [global capacity curve](q4-capacity-curve.md) fits one cache across the ordered five-prompt prefill workload. At eight slots over the 21 CPU-resident target layers, global static reaches 33.13% and LRU reaches 35.13%; static overtakes at 16 slots. The global result supersedes the earlier single-prompt static-8 recommendation.

## Current decision

Use the verified b10002 CUDA build for the measured inference/memory baseline and the verified b10002 Vulkan build for transparent router telemetry. Keep the backend identity attached to every artifact and use fixed-prompt prefill when comparing them.

This recovers a parity-safe GPU trace path, but it does not authorize live caching. The later [expert-size and transfer checkpoint](q4-expert-transfer.md) measures the remaining Layer 0 inputs; per-layer deadlines and end-to-end savings are still unmeasured, and policy hit rates remain estimates rather than latency savings.
