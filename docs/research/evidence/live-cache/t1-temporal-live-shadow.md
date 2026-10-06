# T1 temporal live-shadow result

## Verdict

**T1 runtime correctness and timing measurement: PASS.**

**Original eviction-coupled T2 policy: NO-GO.**

**Bounded two-slot sidecar T2 experiment: AUTHORIZED.**

The frozen temporal predictor reproduces exactly in the live runtime, preserves
tokens and routing, and finishes far before the next layer-24 demand. The
specific highest-ranked-missing-candidate admission rule is nevertheless
overwhelmingly wasteful on the approved focused live suite: only 18 of 126
admissions were useful, while 108 were unused and seven introduced simulated
eviction regret. That rejects speculative admission into the protected 32-slot
LRU. It does not independently reject a bounded sidecar experiment whose two
speculative slots cannot evict normal reactive residents.

This is a shadow-only measurement. `live_cache_enabled=false`; no cache mapping
was changed and no expert weights were transferred.

## Frozen configuration

- Relationship: layer-24 decode experts at token `t` to layer-24 decode experts
  at token `t+1`
- Artifact: `expertflow-temporal-layer24-v1.bin`
- Weights: transition 0.5, current-set retention 0.4, session frequency 0.1
- Candidate width: 16
- Cache model used only for offline admission analysis: reactive LRU, 32 slots
- Runtime: one canonical T1 probe binary, `-ngl 10`, 12 threads, greedy decode
- Workloads: general, code, and translation
- Repetitions: one warmup plus three measured pairs per workload
- Generated tokens: 16 per run
- Measured transitions: 126

The first decode token seeds state and emits no prediction. Every recorded join
uses consecutive forward and decode indices. Each fresh process resets the
conversation generation to one, produces 14 transitions, and retains exactly
one final pending prediction for the unobserved next token.

## Exactness

- Prompt-token parity: exact for all 12 disabled/enabled pairs
- Generated-token parity: exact for all 12 pairs
- Router IDs, order, and recorded weights: exact
- Offline/live candidate IDs and order: exact
- Offline/live floating-point scores: exact
- Offline/live session-frequency state: exact
- Transition counts and causal ordering: exact
- Native golden-vector test: pass

The runtime artifact is binary, versioned, hashed, and fail-closed. It validates
model, runtime, manifest, configuration, payload dimensions, and payload
checksum before enabling T1.

## Live lead-time measurement

Host monotonic timestamps were recorded at source routing observation,
predictor completion, and the next token's layer-24 demand. They do not claim
CUDA overlap.

| Metric | Minimum | p5 | p50 | p95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Predictor latency | 9.5 us | 10.2 us | 11.8 us | 17.5 us | 190.7 us |
| Predictor-finished to next demand | 35.120 ms | 36.064 ms | 38.421 ms | 42.864 ms | 51.829 ms |

All 126 transitions exceed each reference deadline:

- measured 3,345,412-byte CUDA-event H2D reference: 0.306016 ms;
- host staging plus H2D: 1.331416 ms;
- staging, 0.0222 ms queue reference, H2D, and 0.250 ms safety margin:
  1.603616 ms.

Per-domain median lead time was 39.441 ms for general, 38.041 ms for code, and
38.386 ms for translation. Timing therefore supports a transfer opportunity.
Candidate quality remains the main risk for the bounded sidecar experiment.

## Ranking and exact one-transfer rule

Aggregate live ranking:

- Hit@1: 71.43%
- Hit@2: 78.57%
- Hit@4: 83.33%
- Recall@8: 50.00%
- Recall@12: 61.90%
- Recall@16: 68.75%
- Mean reciprocal rank: 0.7861
- Mean predicted candidates already resident: 12.78 of 16

The exact intended T2 rule selected the highest-ranked missing candidate from
the simulated 32-slot reactive cache:

- admissions: 126
- useful and estimated ready: 18 (14.29%)
- wasted: 108 (85.71%)
- no-admission cases: 0
- estimated late-useful: 0
- potential blocking misses avoided: 18
- eviction-regret cases: 7
- mean chosen rank: 11.10

The result is strongly domain-dependent:

| Domain | Useful | Wasted | Useful rate | Eviction regret |
|---|---:|---:|---:|---:|
| General | 12 | 30 | 28.57% | 4 |
| Code | 3 | 39 | 7.14% | 2 |
| Translation | 2 | 40 | 4.76% | 0 |

One useful count is not attributable after summing the independently restarted
domain simulations because aggregate cache state is continuous across the
concatenated analysis. The authoritative aggregate is 18 useful / 108 wasted;
per-domain rows are diagnostic rather than additive.

## Overhead and memory

Measured mean temporal-versus-disabled deltas:

- decode TPS: **+1.44%**
- prompt TPS: **-0.41%**
- end-to-end time: **-0.01%**
- time to first token: **+0.37%**
- process peak private memory: **+6.29 MiB**
- process peak working set: **+0.12 MiB**
- system-wide peak GPU use: **+3.22 MiB**

The small paired performance changes are within run variance and do not support
a speedup claim. Exact native storage sizes are:

- temporal state: 1,304 bytes
- predictor artifact object: 131,216 bytes
- one transition record: 840 bytes
- preallocated record capacity: 8,192
- record storage: 6,881,280 bytes

All 24 focused processes returned zero and exited. GPU sampling is system-wide,
not process-owned; settled before/after noise reached 27 MiB on one run, with
no surviving router or llama process and no cache allocation.

## Verification and identities

- ExpertFlow tests: 169 passed
- Native temporal golden test: passed
- Judge replay: 8 events, 64 demands, 26 static-hotset hits, 19 LRU hits
- `git diff --check`: passed in both worktrees
- Focused summary SHA-256:
  `aadebc790efd4300d9ecc8eb4831e375b4d671244752bd5a95d021ecd9c01556`
- Temporal artifact SHA-256:
  `4ad1c9e991be050d582016c8bce4fd7c7480138e1ed49154657bc0cc2e9e9200`
- T1 probe SHA-256:
  `6b855d9cdd41713991bb79a80b6e06a747475168665e43241c561b6c0492efbc`
- T1 `ggml-base.dll` SHA-256:
  `824f024534850e6974dc32b47430d33c28a0f5006003c74b29bacca4361680ef`
- Judge replay SHA-256:
  `40cd90a1bc45f3e65c4293eb41c0646d248bc86dbf7e4223d8efb19804168139`

Raw evidence is under
`C:\models\expertflow\runs\t1-temporal-live-shadow\focused-final`.

## Decision

Do not implement T2 by admitting predictions into or evicting from the normal
32-slot LRU. The empirical foundation is sound: exact causal temporal
prediction is real and the physical lead-time window is generous. The current
frozen scorer, however, ranks a missing expert that is actually needed next
only 14.29% of the time and performs especially poorly on code and
translation.

A revised experiment may use exactly two dedicated speculative ping-pong slots
outside the normal LRU policy. Waste rate alone is not a failure in that
experiment; measured end-to-end overhead, ready-useful completion, blocking
reduction, exactness, and resource stability decide the result.

The core project result remains intact: ExpertFlow now separates prediction
quality from transfer feasibility with exact runtime evidence. T1 does not
permit retuning on this focused suite. The separately authorized T2 sidecar
stage remains limited to one transfer per token, one layer, and two dedicated
slots.
