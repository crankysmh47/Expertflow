# Stratified Q4 cache-capacity curve

Date: 2026-07-15 PKT

This checkpoint combines only the fixed-prompt prefill portion of the five parity-safe Vulkan traces. It fits one global policy over the ordered workload rather than selecting a different hotset for each prompt. Generated tokens are excluded so the workload is identical across the CUDA and Vulkan evidence paths.

- Events: 3,840 across all 30 MoE layers
- Expert demands: 30,720
- Target subset: layers 0–20, the 21 repeating layers left on CPU by the measured 10-layer CUDA offload
- Target-subset events: 2,688
- Target-subset demands: 21,504
- Projected aligned cache slot: 3,346,048 bytes
- Measured pinned H2D mean for two weight slices: 0.235042 ms/expert
- Structured result: `C:\models\expertflow\runs\stratified-q4-vulkan\capacity-curve-cpu21.json`, 8,268 bytes, SHA-256 `d4922e70c320265b6ddf1e7f0a3ea4a50617bd88921835118c2cd890200b9b4d`

## Global target-layer curve

| Slots/layer | Projected cache | Static hit rate | LRU hit rate | Static serialized H2D/token | LRU serialized H2D/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 536.09 MiB | 33.13% | 35.13% | 26.40 ms | 25.61 ms |
| 16 | 1,072.19 MiB | 50.79% | 48.86% | 19.43 ms | 20.20 ms |
| 32 | 2,144.38 MiB | 72.13% | 68.37% | 11.01 ms | 12.49 ms |
| 64 | 4,288.76 MiB | 92.77% | 84.55% | 2.85 ms | 6.10 ms |
| 75 | 5,025.89 MiB | 96.28% | 88.03% | 1.47 ms | 4.73 ms |
| 96 | 6,433.14 MiB | 99.57% | 90.17% | 0.17 ms | 3.88 ms |
| 108 | 7,237.28 MiB | 100.00% | 90.31% | 0.00 ms | 3.82 ms |

The measured configurable headroom is 7,234 MiB after a separate 1,024 MiB safety reserve, so 108 slots/layer is already 3.28 MiB over budget before cache-specific workspace. Capacities through 96 fit the measured envelope; that does not make 96 the automatic recommendation.

The H2D columns are estimates: mean miss count × the measured per-expert pinned transfer mean, serialized across the 21 selected layers. They omit overlap, contention, launch/scheduling work, cache transitions, and compute deadlines. They are not end-to-end latency predictions.

## Why this supersedes the static-8 conclusion

The earlier 40.68% static-8 figure is a demand-weighted aggregate of five independently optimized prompt-local hotsets. That is useful as a per-prompt upper bound, but it is not one deployable global cache. A single global static-8 cache reaches 33.93% over all 30 layers and 33.13% over the 21 target layers. Global LRU reaches 35.13% on the target layers and therefore wins at the eight-slot budget.

Static placement overtakes LRU at 16 slots and becomes materially stronger at 64 slots. However, all static hotsets in this checkpoint are fit and evaluated on the same five public synthetic prompts. Their high-capacity hit rates are in-sample feasibility evidence, not held-out generalization.

## Decision

The old single-prompt static-8 machine recommendation is superseded for the stratified workload. The regenerated recommendation selects static-96 as the largest tested point that fits the measured envelope, but retains `live_cache_enabled=false` and carries forward these blockers:

- no held-out workload for static-hotset selection;
- no measured per-layer CUDA compute deadlines;
- no live cache or exact same-runtime end-to-end comparison.
