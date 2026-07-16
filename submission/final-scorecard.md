# Final scorecard

## Submission outcome

Official demo: Observatory.

Reason: the exact predictive runtime did not beat the same-runtime reactive
baseline. The Build Week selection gate therefore chooses the Observatory
story, not a hybrid speedup story.

## Observatory proof

| Item | Result | Label |
|---|---:|---|
| Model artifact | 14,439,361,440 bytes | measured |
| Layer-expert objects | 3,840 | measured |
| Aligned Q4 expert slot | 3,346,048 bytes | measured |
| Static-96 held-out hit rate | 87.57% | estimated policy result on measured held-out traces |
| Conversation-reset LRU hit rate | 86.34% | estimated policy result on measured held-out traces |
| Aligned pinned H2D p50 | 0.234016 ms | standalone CUDA measured |
| Aligned pinned H2D p95 | 0.234272 ms | standalone CUDA measured |
| Static-96 cache allocation | 6,433.14 MiB | projected from exact packed bytes |
| Configurable reserve | 800.86 MiB | projected in the frozen measured envelope |
| Judge fixture | 8 events / 64 demands | previously measured trace |
| Fixture static / LRU hits | 26 / 19 | estimated replay |

## Final predictive runtime appendix

This table is evidence about the last bounded experiment. It is not the
submission's product-performance claim.

| Result | Reactive baseline | Final predictive | Outcome |
|---|---:|---:|---:|
| Decode TPS | 27.8598 | 27.5374 | -1.15% |
| Prompt TPS | 21.4306 | 21.3866 | -0.23% |
| End-to-end time | 2916.67 ms | 2927.63 ms | 0.43% slower |
| p95 token latency | 40.855 ms | 41.746 ms | 2.18% worse |
| Blocking transfer time | 163.669 ms | 165.254 ms | 0.97% worse |
| Ready-useful prefetches | 0 | 1.0 mean/run | exact |
| Actual misses prevented | 0 | 1.0 mean/run | 3 per general run; 0 code/translation |
| Total expert bytes moved | 594.37 MB | 643.43 MB | 8.26% amplification |
| Peak system GPU use | 6534.44 MiB | 6544.67 MiB | +10.22 MiB |
| Remaining system VRAM at sampled peak | 9776.56 MiB | 9766.33 MiB | system-wide, not process-owned |
| Token/router parity | pass | pass | exact |

The predictive mode moved 50,181,180 speculative bytes per run. Mean wasted
bytes were 43,490,356. CUDA-event H2D time was 4.051 ms in aggregate across 15
transfers. The host queue-to-ready interval is reported separately and is not a
CUDA latency or overlap claim.

## Strongest runtime result

The exact layer-24 32-slot reactive cache remains the strongest ExpertFlow
runtime milestone:

- 57.33% fewer misses than the eight-slot cache;
- 45.45% lower measured layer-24 blocking wall time;
- 9.75% higher decode TPS than the matched observer/cache-off runtime.

It still reached only 27.71 decode TPS versus 98.53 for the strongest stable
stock full-offload configuration. That gap is why the release does not claim a
lower hardware boundary or an end-to-end runtime win.

## Release defaults

- `live_cache_enabled=false`
- predictive sidecar disabled
- no MTP, RL, or multi-layer prefetch
- no runtime speedup claim
