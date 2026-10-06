# T2 Temporal Sidecar Result

## Verdict

T2 is an exact, stable, negative performance result.

The bounded 34-slot layer-24 sidecar proved that the unchanged packed
`MUL_MAT_ID` operation can consume a mixed mapping from the protected 32-slot
reactive arena and dedicated physical slots 32-33. All measured runs preserved
prompt tokens, generated tokens, router expert IDs, ordering, event counts, and
determinism. No routing weights were present in the canonical trace, so no
separate weight-parity claim is made.

The sidecar did not improve end-to-end throughput. Across the nine measured
general, code, and translation pairs, decode TPS decreased 0.90%, prompt TPS
decreased 0.23%, and end-to-end time increased 0.41%. Blocking time changed by
only +0.26% in the favorable direction and reactive miss count did not change.
No speedup is claimed.

The decisive diagnosis is that all 33 ready-useful sidecar demands were already
reactive hits in the paired baseline. They were selected while absent before
the source token's reactive admission, then entered slots 0-31 during that
source token. The asynchronous transfers were ready, exact, and safe, but none
covered a baseline blocking miss.

## Frozen scope

- model: Gemma 4 26B A4B Q4_0
- runtime: pinned ExpertFlow llama.cpp branch
- offload: `-ngl 10`
- layer: 24 only
- reactive slots: 0-31
- speculative slots: 32-33
- transfer budget: at most one asynchronous prediction per decode token
- predictor: frozen temporal 0.5/0.4/0.1 scorer, width 16
- fallback: unchanged exact blocking reactive cache
- promotion: disabled
- default: `EXPERTFLOW_T2_TEMPORAL_SIDECAR` is off

No CUDA kernel, GGML operation, graph placement, backend split, quantization,
or expert repacking change was made.

## Correctness ladder

The assertion-active native tests cover the sidecar state machine, ping-pong
selection, ready-useful, late-useful, incomplete waste, expiry, release,
generation ownership, stale identity rejection, explicit bounds, conditional
34-slot layout, and external mapping that never enters reactive state.

The live smoke produced one ready-useful sidecar execution with exact
reactive/T2 token and router parity. The full matrix used general, code, and
translation prompts with one warmup plus three measured repetitions per domain.
All 12 pairs passed exact token and router parity. Within each mode and domain,
tokens and router projections were deterministic across all four repetitions.

Prompt/generated token counts were:

| Domain | Prompt tokens | Generated tokens |
|---|---:|---:|
| General | 42 | 16 |
| Code | 53 | 16 |
| Translation | 56 | 16 |

## Measured performance

The table reports means across the nine measured pairs. Variance and every
repetition-level value are preserved in
`t2-temporal-sidecar-summary.json`.

| Metric | Reactive C5 | T2 sidecar | Relative result |
|---|---:|---:|---:|
| Prompt TPS | 20.8796 | 20.8250 | -0.23% |
| Decode TPS | 27.1458 | 26.9018 | -0.90% |
| End-to-end time | 2995.86 ms | 3009.16 ms | +0.41% slower |
| TTFT | 2407.14 ms | 2415.05 ms | +0.33% slower |
| Token latency p50 | 39.142 ms | 39.418 ms | +0.70% |
| Token latency p95 | 42.409 ms | 42.491 ms | +0.19% |
| Peak GPU memory | 6515.33 MiB | 6525.44 MiB | +10.11 MiB |
| Reactive blocking misses | 177.67 | 177.67 | 0% reduction |
| Total blocking time | 170.264 ms | 169.741 ms | 0.26% reduction |

Per-domain decode TPS changes were -0.55% general, -1.59% code, and -0.55%
translation. The code-domain third measured pair was the largest regression at
-3.68%; the complete repetition values and basic variance remain in the JSON
evidence rather than being hidden by the aggregate.

## Transfer behavior

Each measured T2 run enqueued exactly 15 transfers:

- mean ready-useful: 3.67
- mean late-useful: 0
- mean wasted: 10.33
- one final transfer per run had no target demand because generation ended
- transferred bytes: 50,181,180 per run
- wasted bytes: 34,569,257 mean per run
- sidecar blocking wait: 0 ms
- aggregate CUDA-event H2D: 4.094 ms mean per run
- aggregate staging: 4.722 ms mean per run
- aggregate enqueue: 0.295 ms mean per run
- aggregate queue-to-ready host interval: 566.431 ms mean per run

The queue-to-ready value is an aggregate host interval across 15 transfers. It
is not CUDA-event latency and does not claim copy/compute overlap. Mean
CUDA-event H2D was approximately 0.273 ms per transfer.

## Memory and cleanup

The projected 34-slot arena was 113,744,128 bytes. The backend-measured
allocation was 113,744,640 bytes, 512 bytes higher because of actual buffer
allocation alignment. The allocation was constant across every measured T2
run.

All 24 focused processes exited. Settled GPU memory showed no persistent growth
trend; the maximum single-run settled delta was 10 MiB in the warmup and later
runs returned within the normal background-process variation. No stale slot,
generation mismatch, invalid value, crash, or corruption was observed.

## Verification

- ExpertFlow: 176 tests passed
- native cache test: passed
- native sidecar test: passed
- native temporal test with frozen artifact: passed
- judge replay: 8 events, 64 demands, 26 static-hotset hits, 19 LRU hits
- judge replay SHA-256:
  `40cd90a1bc45f3e65c4293eb41c0646d248bc86dbf7e4223d8efb19804168139`
- exact cross-repetition token/router determinism: passed
- live cache and T2 remain disabled by default

The first suite invocation and first full pytest invocation failed before
inference/tests because of missing Python namespace paths. Both environment
failures and their corrected commands are preserved in the append-only ledger.

## Artifacts

- concise measured summary:
  `docs/evidence/live-cache/t2-temporal-sidecar-summary.json`
- full raw runs and ledger:
  `C:\models\expertflow\runs\t2-temporal-sidecar\focused`
- runtime:
  `C:\models\expertflow\runs\t2-temporal-sidecar\runtime`
- measured-summary SHA-256:
  `adef89285ddfb6f528c63ef3882283405174c31e865eb16581d7abd4aa312b09`
- append-only ledger SHA-256:
  `295be1cb9146c2f7027041fa73fa8d0e7811285001f9d592e096d4f87a0f4611`
- `llama.dll` SHA-256:
  `f8489344b5a6a5ae3abdb23dd84161df238915848b8efe6a579970cd367cb7af`
- `ggml-base.dll` SHA-256:
  `18a26438a46cb653b97f90df181509f4ce4c01e7e58bf88488a039b41396219a`
- `ggml-cuda.dll` SHA-256:
  `302096f1517e2b46bfdb8b78efa8c83688b5c51e83dceb460a5b1657129fea3e`
- router probe SHA-256:
  `b0d6c6f708b3310bf234cbed4137d922e0385b9d19958c9e32b5dc6b53dc025c`

Toolchain: NVIDIA driver 591.86, CUDA 12.8, MSVC 19.39.33523, CMake
4.3.1, Ninja 1.13.2, RTX 5060 Ti 16,311 MiB.

## Next decision

The user-authorized prerequisite for considering a later bounded gating step is
met: unconditional T2 is exact but measurably slower. No such expansion was
started here.

The strongest next hypothesis is not wider concurrency or multi-layer
prediction. It is a narrow causal admission filter that evaluates candidates
against the projected post-source-token reactive state, so a transfer targets
an expert that would still be absent after the current true demands are
admitted. That must be specified and approved separately. Confidence gating,
wider transfer concurrency, and multi-layer temporal prediction remain closed.
