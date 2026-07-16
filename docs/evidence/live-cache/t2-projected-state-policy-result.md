# Final Projected-State Temporal Policy Result

## Verdict

The final bounded predictive-policy experiment is exact and stable, and it
prevents real blocking misses, but it does not improve throughput.

The projected-state filter fixed the causal flaw in T2. Before choosing a
candidate, it copies the layer-24 reactive cache state and applies the current
token's authoritative admissions, evictions, and LRU update. It then prefetches
the highest-ranked frozen temporal candidate that remains absent. Prediction
does not mutate the live reactive state.

Across nine measured reactive/predictive pairs, the policy prevented three
actual paired-baseline blocking misses in every general-chat run and zero in
code and translation. No ready-useful sidecar demand was a reactive-baseline
hit. Despite that improvement, decode TPS decreased 1.15%, prompt TPS decreased
0.23%, end-to-end time increased 0.43%, and measured total blocking time
increased 0.97%. No speedup is claimed.

This is the final runtime experiment. ExpertFlow runtime policy, cache size,
layer scope, predictor, and transfer concurrency are frozen. No additional
predictor, multi-layer prefetch, MTP, RL, or cache-size exploration is
authorized for the release.

## Frozen scope

- model: Gemma 4 26B A4B Q4_0
- offload: `-ngl 10`
- cache layer: 24 only
- reactive slots: 0-31
- protected speculative slots: 32-33
- exact arena allocation: 113,744,640 bytes
- predictor: frozen temporal 0.5/0.4/0.1 scorer, width 16
- transfer budget: one asynchronous transfer per decode token
- fallback: unchanged exact blocking reactive cache
- promotion: disabled
- default: `EXPERTFLOW_T2_TEMPORAL_SIDECAR` and live caching are off

No CUDA kernel, GGML operation, graph partition, backend placement,
quantization, repacking, or allocator redesign was introduced.

## Correctness

The matrix used general, code, and translation prompts with one warmup plus
three measured repetitions per domain. All 12 reactive/predictive pairs
preserved exact prompt tokens, generated tokens, router expert IDs, expert
ordering, event counts, and causal identity. Routing weights were not present
in the canonical trace, so no separate weight-parity claim is made.

Within each domain and mode, measured repetitions produced identical tokens and
router projections.

| Domain | Prompt tokens | Generated tokens |
|---|---:|---:|
| General | 42 | 16 |
| Code | 53 | 16 |
| Translation | 56 | 16 |

## Measured performance

Means below cover the nine measured pairs. Repetition values and basic variance
are preserved in `t2-projected-state-policy-summary.json`.

| Metric | Reactive C5 | Final predictive | Relative result |
|---|---:|---:|---:|
| Prompt TPS | 21.4306 | 21.3866 | -0.23% |
| Decode TPS | 27.8598 | 27.5374 | -1.15% |
| End-to-end time | 2916.67 ms | 2927.63 ms | +0.43% slower |
| TTFT | 2342.73 ms | 2346.91 ms | +0.18% slower |
| Token latency p50 | 38.095 ms | 38.356 ms | +0.68% |
| Token latency p95 | 40.855 ms | 41.746 ms | +2.18% |
| Peak GPU memory | 6534.44 MiB | 6544.67 MiB | +10.22 MiB |
| Blocking misses | 177.67 | 177.33 | 0.21% reduction |
| Total blocking time | 163.669 ms | 165.254 ms | 0.97% worse |

Per-domain decode changes were -1.66% general, -0.20% code, and -1.60%
translation. General-chat blocking misses fell from 158 to 157 per run because
three sidecar uses replaced paired-baseline misses; the aggregate event count
reports one fewer reactive miss because the three measured repetitions share
the same deterministic trace.

## Prediction and transfer behavior

Each measured predictive run enqueued exactly 15 transfers:

- general: 3 ready-useful, 0 late-useful, 11 wasted, 1 unused at generation end
- code: 0 ready-useful, 0 late-useful, 14 wasted, 1 unused
- translation: 0 ready-useful, 0 late-useful, 14 wasted, 1 unused
- actual paired-baseline blocking misses prevented: 3 per general run, 0
  elsewhere
- ready sidecar demands that were paired-baseline hits: 0
- transferred bytes: 50,181,180 per run
- mean wasted bytes: 43,490,356 per run
- sidecar blocking wait: 0 ms
- aggregate CUDA-event H2D: 4.051 ms mean per run
- aggregate staging: 4.491 ms mean per run
- aggregate enqueue: 0.286 ms mean per run
- aggregate queue-to-ready host interval: 552.160 ms mean per run

The host queue-to-ready interval is not CUDA-event latency and is not a
copy/compute-overlap claim. Mean CUDA-event H2D time was approximately
0.270 ms per transfer.

The filter moved selections deeper into the frozen ranking: measured candidate
ranks ranged from 7 through 14. This confirms that current-token projected
residency was applied rather than retuning the predictor.

## Memory and cleanup

The packed arena remained exactly 113,744,640 bytes in every predictive run.
All 24 focused processes exited. Settled GPU-memory deltas ranged from -4 MiB
to +1 MiB across individual fresh processes, with no persistent growth trend.
No stale generation, invalid mapping, race, corruption, crash, or cleanup
failure was observed.

## Verification

- ExpertFlow: 179 tests passed
- native projected-cache policy test: passed
- native sidecar test: passed
- native predictor test with frozen artifact: passed
- native temporal test with frozen artifact: passed
- judge replay fixture: passed
- judge replay: 8 events, 64 demands, 26 static-hotset hits, 19 LRU hits
- exact cross-repetition token/router determinism: passed
- `git diff --check`: passed
- live cache and predictive sidecar remain disabled by default

Failed commands are preserved rather than hidden: the first build used a
nonexistent conventional Visual Studio path; the first full pytest invocation
omitted `PYTHONPATH=.`; the first native aggregate command omitted required
predictor/temporal artifact arguments; and the first determinism check compared
raw trace hashes containing nonsemantic runtime fields. Corrected commands
passed.

## Artifacts

- summary:
  `docs/evidence/live-cache/t2-projected-state-policy-summary.json`
- full runs and append-only command ledger:
  `C:\models\expertflow\runs\t2-projected-state-policy\focused`
- summary SHA-256:
  `5da7eefb7ad5ed20b48cb69e8ee112b4f2acb7b63b51a35edaadd3a875c25373`
- ledger SHA-256:
  `996700006e268434d4ec6a4dbed9db0cea045459e8e81387d632792a346ceae2`
- `ggml-base.dll` SHA-256:
  `1ab2b52cfce3acafd754c29e1c9447e6401b3cb133ff1e6443a8e23d02f87101`
- `ggml-cuda.dll` SHA-256:
  `302096f1517e2b46bfdb8b78efa8c83688b5c51e83dceb460a5b1657129fea3e`
- `llama.dll` SHA-256:
  `f8489344b5a6a5ae3abdb23dd84161df238915848b8efe6a579970cd367cb7af`
- router probe SHA-256:
  `b0d6c6f708b3310bf234cbed4137d922e0385b9d19958c9e32b5dc6b53dc025c`

Toolchain: NVIDIA driver 591.86, CUDA 12.8, MSVC 19.39.33523, CMake
4.3.1, Ninja 1.13.2, RTX 5060 Ti 16,311 MiB.

## Release decision

Freeze the runtime with this result as honest experimental evidence. The
release should lead with the exact Observatory, replayable empirical evidence,
the working exact cache primitive, and the demonstrated causal lesson:
asynchronous copies can become ready and can prevent true misses, but this
small frozen predictor and one-layer sidecar do not yield end-to-end speedup.

Release integration, final benchmark curation, documentation, and submission
assets may proceed. Runtime R&D is closed.
