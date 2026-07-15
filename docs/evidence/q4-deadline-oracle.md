# Decode transfer deadline and oracle evidence

> **Quarantined historical evidence:** This result derives from the callback now labeled `trace_v1_perturbing`. It is retained for audit but excluded from final locality, deadline, policy, recommendation, and Gate 4 claims. The `93.28%` value is withdrawn pending a parity-safe replacement corpus. See `configs/trace-evidence-status.json`.

Date: 2026-07-15 PKT

This checkpoint evaluates the deployment-relevant cross-phase policy: fit static residents on the original five prompt-prefill traces, then score the five untouched held-out decode traces. It also compares measured Vulkan callback windows with the CUDA transfer microbenchmark as an explicitly backend-specific oracle bound.

- Training: 2,688 prefill events over target layers 0–20
- Evaluation: 735 decode events, 5,880 expert demands, and 35 complete decode forwards
- Cache: static-96, 6,433.14 MiB projected across 21 layers
- Transfer input: 0.235042 ms per cold expert from pinned host memory
- Timing-window source: parity-safe b10002 Vulkan evaluation callbacks
- Gate: CONDITIONAL; live cache disabled

## Cross-phase held-out curve

| Slots/layer | Frozen static hit rate | Cold-start LRU hit rate | Static serialized H2D/token | LRU serialized H2D/token |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 27.89% | 34.73% | 28.47 ms | 25.77 ms |
| 16 | 43.64% | 50.83% | 22.26 ms | 19.41 ms |
| 32 | 61.68% | 67.94% | 15.13 ms | 12.66 ms |
| 64 | 81.67% | 76.82% | 7.24 ms | 9.15 ms |
| 75 | 86.24% | 76.97% | 5.43 ms | 9.09 ms |
| 96 | 93.28% | 76.99% | 2.65 ms | 9.09 ms |
| 108 | 94.57% | 76.99% | 2.14 ms | 9.09 ms |

Static-96 remains the highest tested point inside the measured memory envelope. Its held-out decode hit rate is lower than the 96.45% held-out prefill rate, so the decode result is used for the current recommendation.

## Observed layer windows

Within each complete held-out decode forward, the parity-safe Vulkan trace exposes 700 adjacent target-layer windows:

- minimum: 0.5617 ms
- median: 1.4955 ms
- mean: approximately 1.49 ms
- p95: 1.9305 ms
- maximum: 2.8242 ms

These are callback-observed Vulkan intervals. They include the instrumented backend's scheduling and callback behavior and are not CUDA kernel deadlines.

## One-layer oracle bound

At static-96, the held-out decode workload has 395 cold expert selections. A reactive/static cache with no prefetch would serialize approximately 2.6526 ms of pinned H2D transfer per token.

The oracle test assumes perfect knowledge of the next layer's missing experts at the preceding layer callback and permits their transfers to use the observed interval. Under that non-deployable assumption:

- 719 of 735 layer events are ready by the observed deadline;
- 16 events remain late;
- every late event is layer 0, which has no preceding target-layer window;
- estimated residual blocking falls to 0.2485 ms/token.

This is evidence that one-layer lookahead could be valuable, not evidence that a predictor can achieve it. It also combines Vulkan timing windows with a standalone CUDA transfer curve and does not measure copy/compute contention.

## Artifacts

- Cross-phase curve: `C:\models\expertflow\runs\heldout-q4-vulkan\prefill-train-decode-eval-cpu21.json`, 8,846 bytes, SHA-256 `6d60685498e31edcb56ff2a90e112f04130c35c189b27bf02a9bfaac7e2c52f1`
- Oracle timeline: `C:\models\expertflow\runs\heldout-q4-vulkan\oracle-deadline-vulkan-cpu21.json`, 248,389 bytes, SHA-256 `627de780f3cc501caf6041ae6d029063fe90061be8ba845ae8fe27b8fde65c45`
- Decode recommendation: `C:\models\expertflow\runs\q4-probe\recommendation-decode-heldout.json`, 1,824 bytes, SHA-256 `10b4d3bafe1295182915adc5a6415b798742cfeeac73c21c98dca6fbc00162d2`
- Decode replay: `C:\models\expertflow\runs\q4-probe\report-decode-heldout.html`, 51,048 bytes, SHA-256 `58944c45a978fb16aba777e8453def7b4ce5a2283ae8e54901c66628aa04b16b`

The decode replay uses prefill training traces to freeze residents and held-out decode traces for evaluation. It reconciles to 5,485 ready and 395 blocking selections and keeps training/evaluation phases explicit in its reproduction command.

## Decision

Use static-96 as the Observatory replay configuration for held-out decode. Do not enable a live cache. The Vulkan oracle makes the next engineering question concrete, but it does not close either remaining gate:

- CUDA per-layer deadlines and copy/compute contention are not measured;
- no same-runtime live cache exists for an exact end-to-end comparison.
