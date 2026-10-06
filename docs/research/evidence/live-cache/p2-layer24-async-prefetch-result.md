# P2 layer-24 asynchronous prefetch result

## Verdict

**STOP - EXACT BOUNDED NEGATIVE RESULT.**

P2.0 proved a genuine dedicated-stream packed-Q4 H2D transfer and P2.1 proved
deterministic no-mutation admission planning. P2.2 preserved exact output while
issuing at most one predicted transfer per layer-23 to layer-24 transition, but
it produced zero ready-and-useful prefetches in the repeated focused suite and
did not improve end-to-end decode throughput.

P2.3 concurrency expansion is rejected. No multi-layer prediction, higher
`-ngl`, 64-slot cache, MTP, retraining, graph relocation, or allocator redesign
was attempted. The accepted P1 and C5 commits remain the runtime floor.
`live_cache_enabled=false` remains the default.

## Protected ancestry and isolation

- ExpertFlow P1 head: `b50ea6a`
- llama.cpp P1 head: `6e7bdffe`
- C5 is already an ancestor of both P1 heads:
  - ExpertFlow merge base: `0eb05daf`
  - llama.cpp merge base: `641f5313`
- ExpertFlow worktree:
  `C:\models\expertflow\worktrees\p2-layer24-async-prefetch`
- llama.cpp worktree:
  `C:\models\expertflow\worktrees\llama-p2-layer24-async-prefetch`
- No merge or push was performed.

## Pre-change baseline

Before runtime source changes:

- 139 ExpertFlow tests passed.
- Native cache and predictor tests passed.
- Nine focused P1 pairs passed exact prompt/generated-token parity, exact
  ordered router parity, and exact offline/live candidate and score
  equivalence.
- Nine C5 cache-off/cache-on pairs passed exact token and router parity.
- The P1 binary and artifact hashes matched the accepted milestone.

Two invalid tooling attempts are preserved:

1. Python 3.12 validation produced 1-ULP offline score differences because its
   `sum()` accumulation differs from the accepted Python 3.11 environment.
   Live candidates, scores, tokens, binary hash, and artifact hash matched.
   Validation was rerun with pinned Python 3.11 and passed.
2. The first C5 runner invocation omitted CUDA 12.8 from `PATH` and exited with
   Windows loader status `0xC0000135` before model load. The corrected inherited
   environment passed.

## P2.0 dedicated-stream transfer proof

The CUDA backend exposes a narrow opaque transfer service through backend
registry function pointers. It owns:

- one dedicated `cudaStreamNonBlocking` stream
- fixed descriptor storage
- fixed `cudaHostAlloc` pinned staging
- start/end/completion events
- no per-transfer GPU or pinned-host allocation

One known layer-24 expert was transferred into isolated slot 31. The slot was
not exposed to execution or cache residency. Reconciliation occurred on the
next invocation, after an intervening compute window, and copied bytes were
read back and compared with the three packed source ranges.

| Measurement | Result |
| --- | ---: |
| Expert | 44 |
| Packed bytes | 3,345,412 |
| Pageable-to-pinned staging | 1.0254 ms |
| Host enqueue work | 22.2 us |
| CPU return including staging | 1.0476 ms |
| H2D CUDA-event duration | 0.306016 ms |
| Observed enqueue-to-reconcile compute window | 70.375 ms |
| Ready at reconciliation | yes |
| Byte verification | exact |
| Token/router parity | exact |

`queue_to_ready_ns` is host-observed at the first later query. It is not
reported as CUDA-event latency or precise copy/compute overlap.

## P2.1 plan-only result

The predictor and C5 cache were enabled together only under explicit P2 mode.
For each layer-23 event the runtime built one immutable admission decision from
the current layer-24 cache state.

- 57 transitions in the first live check
- exact prompt/generated tokens
- exact ordered router selections
- exact offline/live predictor output
- deterministic plan records
- zero slot mutations
- zero predicted transfers

## P2.2 one-transfer result

The focused comparison used the same runtime, layer 24, 32 slots, `-ngl 10`,
model, general/code/translation prompts, one warmup, and three measured
repetitions per mode. All nine plan and predictive runs matched reactive C5
prompt tokens, generated tokens, and ordered router events exactly.

### Aggregate performance

| Mode | Prompt TPS | Decode TPS | End-to-end |
| --- | ---: | ---: | ---: |
| Reactive C5 | 20.415 | 26.276 | 3,071.950 ms |
| P2.1 plan-only | 20.436 | 26.162 | 3,070.464 ms |
| P2.2 prefetch-one | 20.353 | 26.036 | 3,082.499 ms |

P2.2 versus reactive C5:

- prompt TPS: **-0.30%**
- decode TPS: **-0.91%**
- end-to-end time: **+0.34%**

### Cache and transfer behavior

| Measurement | Reactive C5 | P2.2 |
| --- | ---: | ---: |
| Mean reactive hits/run | 345.0 | 367.3 |
| Mean reactive misses/run | 177.7 | 155.3 |
| Mean reactive blocking/run | 171.422 ms | 152.279 ms |

- reactive miss reduction: **12.57%**
- reactive cache blocking reduction: **11.17%**
- prediction records: 588
- transfers enqueued: 564
- useful at the authoritative target: 228
- ready and useful at the authoritative target: **0**
- late useful transfers: **228**
- unused transfers: 336
- total prefetched bytes: 1,886,812,368
- useful bytes: 762,753,936
- wasted bytes: 1,124,058,432
- exact-fallback event wait: 147.299 ms total, 16.366 ms/run

Transfer timing across 564 operations:

| Timing | p50 | p95 | Mean |
| --- | ---: | ---: | ---: |
| Pageable-to-pinned staging | 0.160 ms | 1.065 ms | 0.336 ms |
| Host enqueue | 17.7 us | 31.84 us | 19.67 us |
| CPU return including staging | 0.179 ms | 1.091 ms | 0.356 ms |
| H2D CUDA event | 0.236 ms | 0.239 ms | 0.246 ms |
| First-query queue-to-ready host wall | 0.825 ms | 1.108 ms | 0.876 ms |

The layer-23 to layer-24 live window is too short for the current staging plus
H2D path. Miss reduction is real, but the useful copies arrive late and exact
fallback absorbs their completion cost. Increasing concurrency would increase
wasted traffic and cannot repair the absence of a ready useful transfer under
this boundary.

## Verification

- 139 ExpertFlow tests passed.
- Focused performance-runner tests passed.
- Native cache state-machine tests passed with assertions active.
- Native predictor tests passed with the frozen artifact.
- Judge replay reproduced 8 events, 64 demands, 26 static-hotset hits, and
  19 LRU hits.
- `git diff --check` passed in both worktrees.
- No model/probe process remained after validation.
- The experimental runtime binary and DLL hashes are preserved with raw
  measurements under
  `C:\models\expertflow\runs\p2-layer24-async-prefetch`.

The experimental source remains isolated and uncommitted because the accepted
P2 passing criterion required a measured end-to-end improvement over identical
reactive C5. That criterion did not pass.

## Recommendation

Do not expand this live boundary. If work resumes after submission, investigate
an earlier causally valid source signal or eliminate pageable-to-pinned staging
before reconsidering asynchronous prediction. Either change requires a new
design review and fresh exactness gates. It should not be treated as an
automatic continuation of P2.
