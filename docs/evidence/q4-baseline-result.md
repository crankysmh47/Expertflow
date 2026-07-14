# Gemma 4 Q4 baseline and routing result

- **Artifact:** `google/gemma-4-26B-A4B-it-qat-q4_0-gguf@21bfe2a8c89118c9a1a2aa242934fc4d1c0fff15`
- **File:** `gemma-4-26B_q4_0-it.gguf`
- **Bytes:** `14,439,361,440`
- **SHA-256:** `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- **Runtime:** official llama.cpp `b10002` (`a7312ae94`), CUDA 12.4
- **Result:** Q4 baseline PASS; telemetry/parity PASS; live-cache gate CONDITIONAL

## Artifact verification

The repository verifier streamed the complete file, matched its exact length and SHA-256, and returned the verified absolute path. The aria2 control map was absent. A revision-pinned Hugging Face HEAD response independently reported the same length and digest through `X-Linked-Size` and `X-Linked-ETag`.

## Measured baseline

All runs used seed `42`, temperature `0`, context `1024`, 12 CPU threads, one single-turn request, no hidden warmup, and the checked-in prompt. Raw output and manifests are under `C:\models\expertflow\runs`.

| Run | GPU layers | Output limit | Elapsed | Peak GPU used | Peak working set | llama.cpp rate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `q4-cpu-smoke` | 0 | 8 | 23.683 s | 2,935 MiB | 11,348,004,864 B | prompt 3.0 t/s; generation 17.4 t/s |
| `q4-gpu10` | 10 | 32 | 57.791 s | 7,437 MiB | 12,964,139,008 B | prompt 4.6 t/s; generation 5.7 t/s |
| `q4-gpu10-smoke8` | 10 | 8 | 8.527 s | 8,053 MiB | 12,433,321,984 B | prompt 27.0 t/s; generation 26.1 t/s |

The first GPU run included cold CUDA setup and used a different output limit, while the short repeat benefited from process-external driver/kernel caches. These are compatibility and memory observations, not a statistically controlled speedup claim. Every run returned `0`, generated text, and released model memory afterward.

## Real router trace

The finalized probe linked to the same verified CUDA DLLs and used 10 GPU layers. Both parity sides submitted identical tokens one at a time so every callback tensor had an unambiguous absolute token index.

| Evidence | Result |
| --- | --- |
| Prompt tokens | 38, exact match |
| Generated tokens | 8, exact match |
| First generated mismatch | none |
| Decoded tokens represented in trace | 45 |
| MoE layers per decoded token | 30 |
| Selected experts per event | 8 |
| Validated router events | 1,350 |
| Trace bytes | 411,526 |

The eighth sampled output token is not decoded again, so the trace correctly covers 38 prompt tokens plus the first 7 generated tokens: `45 × 30 = 1,350` events.

## Locality and policy evidence

This one-prompt trace is intentionally a feasibility sample, not a workload benchmark.

- Mean adjacent-token expert reuse across layers: `32.59%` (range `23.01%` to `44.03%`).
- Mean token reuse distance across layers: `4.44` tokens.
- Unique experts observed per layer: `65` to `91` of `128`.
- Estimated 8-slot static hotset hit rate: `36.37%`.
- Estimated 8-slot online LRU hit rate: `31.87%`.
- Estimated reactive baseline hit rate: `0%` by definition.

The static hotset beats LRU by 4.50 percentage points on this short trace, which is a useful policy result. The overall live-cache decision remains `CONDITIONAL`: collect a stratified multi-prompt trace and add measured transfer timing before implementing or claiming a predictive runtime speedup.
