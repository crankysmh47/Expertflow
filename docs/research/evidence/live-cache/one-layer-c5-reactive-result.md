# Exact One-Layer 32-Slot C5 Reactive Result

## Verdict

**PASS — BOUNDED EXACT REACTIVE CACHE.** Layer 24 executed directly from one persistent 32-slice packed CUDA operand under deterministic conversation-reset LRU. C5-0 through C5-4 passed. The feature remains disabled by default; this result does not authorize prediction, asynchronous transfer, another layer, MTP/ML, or a runtime-speed claim.

## Physical and policy contract

- layer 24 only; authoritative top-8 logical order remapped to physical slots 0–31
- unchanged CUDA `MUL_MAT_ID`; no eight-slice staging copy, repacking, kernel change, graph rewrite, or general allocator rewrite
- lowest-ID free slot, then unprotected least-recently-used slot with ascending slot-ID tie-break
- missing experts processed in authoritative order; all demanded residents protected during the invocation
- exact measured persistent arena: **107,053,696 bytes (102.094 MiB)**
- conservative aligned projection: 107,073,536 bytes (102.113 MiB)

## Correctness ladder

| Gate | Result | Evidence |
|---|---|---|
| C5-0 compiled/disabled | PASS | seven prompts and 23,880 ordered router events exact against the verified C4 baseline |
| C5-1 passthrough | PASS | same seven-prompt parity; all processes reported 32-slot passthrough |
| C5-2 known-set/direct execution | PASS | 66 cache events; exact tokens and 1,980 router events; physical IDs above seven executed directly |
| C5-3 replacement/forced miss | PASS | 66/66 events forced eight blocking reloads; generations advanced exactly once; exact parity |
| C5-4 arbitrary true-router demand | PASS | three repetitions of seven prompts; exact tokens/routing and deterministic cache state |

Across the three C5-4 repetitions, 2,388 cache events contain 19,104 expert demands: 13,353 hits and 5,751 blocking misses (69.8964% reactive hit rate). Transferred packed bytes reconcile exactly to `5,751 * 3,345,412 = 19,239,464,412`. An independent event replay validated all 2,388 access sequences, protected hits, lowest-free choices, LRU/tie-break evictions, loads, generations, bytes, recency values, and final 32-slot mappings with zero failures.

## Diagnostic timing and memory

Among 2,007 miss-bearing events, host-wall blocking duration is p50 1,630 us and p95 7,655 us, range 444–11,708 us. This includes blocking copies and synchronization; it is not CUDA-event latency, overlap, or speedup evidence.

Settled system GPU-use deltas across 21 focused processes ranged from -20 to +5 MiB with median 0 MiB and no monotonic growth. The process sampler did not provide a usable process-specific GPU peak, so no such claim is made. Every process exited and no probe remained.

## Failed approach preserved

The first fresh C5-0 build used CMake's `GGML_NATIVE=ON` default and AVX-512, unlike the verified C4 AVX2/BMI2/SSE4.2 configuration. With the feature disabled, near-tied router ordering changed and three of seven generated-token arrays diverged. That run is preserved at `c5-0-disabled` and rejected. Reconfiguring to the recorded C4 flags produced `c5-0-disabled-corrected`, which restored exact seven-prompt and 23,880-event parity. This is build-configuration evidence, not a cache failure.

## Verification and identity

- 89 ExpertFlow tests passed
- assertion-active native config test passed
- assertion-active llama cache test passed
- judge replay reproduced 8 events / 64 demands / static 26 / LRU 19
- feature-off restoration passed after C5-4
- runtime probe SHA-256: `e796d0a36595008a16d8ed36ece48b07b9bfeea46d4a539cab9f70cba2f3c677`
- `ggml-base.dll` SHA-256: `6176627d6d4ace19ee2554772f4a908528e7b9d697220339055b144f7a3f3813`
- model SHA-256 remains `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`

Raw artifacts are under `C:\models\expertflow\runs\c5-reactive-cache`. No predictor, expanded-collection, protected Observatory, canonical runtime, or pristine pinned checkout was modified by this track.
