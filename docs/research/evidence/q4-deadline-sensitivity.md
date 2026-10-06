# Q4 transfer and deadline sensitivity

This checkpoint deliberately combines two separate evidence sources without treating them as one runtime measurement:

- Transfer duration: CUDA 12.4, idle default stream, pinned host memory, two real expert weight-slice copies
- Available windows: llama.cpp b10002 Vulkan router-callback timestamps
- Simulator: one-layer perfect-future oracle over held-out decode routing
- Classification: `estimated_cross_backend`
- Copy/compute contention measured: no
- Live runtime measured: no

## Pooled transfer inputs

Three independent microbenchmark processes contributed 600 single-copy observations per payload and memory mode. For the two real weight slices:

| Statistic | Down slice | Gate/up slice | Serialized expert total |
| --- | ---: | ---: | ---: |
| p50 CUDA event | 0.079200 ms | 0.156608 ms | 0.235808 ms |
| p95 CUDA event | 0.079456 ms | 0.156832 ms | 0.236288 ms |

The 4-byte scale arrays can remain resident and are not included in the two-slice transfer. A separately packed aligned-expert copy measured 0.234016 ms p50 and 0.234272 ms p95.

## Aggregate deadline sensitivity

Training residents come from 31 parity-safe training-prefill conversations. Evaluation uses 8 held-out decode conversations, 504 complete forwards, 10,584 target-layer events, and 84,672 expert demands.

| Transfer input | Static hit rate | No-prefetch serialized transfer | Oracle late events | Oracle residual |
| --- | ---: | ---: | ---: | ---: |
| Two-slice p50, 0.235808 ms/expert | 87.57% | 4.9234 ms/token | 212 / 10,584 | 0.1702 ms/token |
| Two-slice p95, 0.236288 ms/expert | 87.57% | 4.9335 ms/token | 212 / 10,584 | 0.1705 ms/token |

At p95, the oracle reduces the transfer-only residual by 96.54% relative to a no-prefetch static-96 serialization. This is strong oracle headroom, but it assumes exact future expert knowledge and cross-backend windows. Of 212 late events, 207 are layer 0, 4 are layer 7, and 1 is layer 15.

## Per-domain p95 result

| Conversation/domain | Static hit rate | No-prefetch transfer | Oracle late events | Oracle residual |
| --- | ---: | ---: | ---: | ---: |
| General chat | 94.12% | 2.3329 ms/token | 12 | 0.0450 ms/token |
| Code | 85.81% | 5.6334 ms/token | 3 | 0.0081 ms/token |
| Math/reasoning | 87.58% | 4.9320 ms/token | 15 | 0.0675 ms/token |
| Translation | 88.69% | 4.4895 ms/token | 35 | 0.2100 ms/token |
| Multilingual | 87.47% | 4.9733 ms/token | 25 | 0.1163 ms/token |
| Long context | 88.51% | 4.5607 ms/token | 36 | 0.2775 ms/token |
| Structured output | 85.80% | 5.6372 ms/token | 46 | 0.2325 ms/token |
| Topic shift | 82.60% | 6.9086 ms/token | 40 | 0.4073 ms/token |

Artifacts:

- p50: `deadline-static96-two-slice-p50.json`, 3,677,285 bytes, SHA-256 `477c3fcb2a8857e52af25a049495a598a516f751a3a32bc4921c9e16f21d9d79`
- p95: `deadline-static96-two-slice-p95.json`, 3,679,388 bytes, SHA-256 `4d46952c1eee603ce6bab09201a75c116f1b493aa061283ba8a9b296a320e8ee`

The oracle result does not clear the live gate. The practical static policy reduces decode cold bytes by only 9.03% versus LRU and loses on three domains. CUDA per-layer compute windows, real copy/compute contention, slot replacement, and same-runtime end-to-end latency remain unmeasured. `live_cache_enabled=false`.
