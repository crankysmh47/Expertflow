# T1 Temporal Live-Shadow Design

## Objective

Integrate the frozen T0 layer-24 next-token predictor into the canonical
observer runtime without changing cache residency or moving expert weights.
T1 must prove exact offline/live prediction equivalence and measure whether the
real interval from token `t` layer-24 routing to token `t+1` layer-24 routing
can accommodate one packed-Q4 transfer.

T0 remains preserved at ExpertFlow commit `afe17d1`. P2 remains preserved at
llama.cpp commit `d8354b17`. T1 is disabled by default and
`live_cache_enabled=false`.

## Chosen architecture

T1 uses a separate temporal artifact and runtime state rather than extending
the accepted P1 adjacent-layer artifact in place. This preserves P1/P2 binary
compatibility and makes temporal decoding fail closed on its own identity,
dimensions, scorer, and reset rules.

The artifact contains:

- magic and format version;
- model, runtime, manifest, and temporal-selection hashes;
- layer 24, eight source experts, and sixteen candidates;
- the 128 by 128 source-normalized temporal transition table;
- frozen weights `0.5 / 0.4 / 0.1`;
- ascending-ID tie-break contract;
- payload length and SHA-256.

No prefill table exists. Loading or calling the scorer for prefill fails.

## Frozen scorer

For each decode layer-24 observation:

1. increment session counts for the eight true current experts;
2. sum their source-normalized transition rows;
3. divide transition scores by the maximum transition score, or one when all
   scores are zero;
4. add `0.4` for experts in the current selected set;
5. add `0.1 * session_count / max_session_count`;
6. retain candidates with finite positive combined score;
7. sort by descending score and ascending expert ID;
8. require at least sixteen supported candidates.

This order exactly matches the frozen offline T0 implementation. Golden
vectors contain source IDs, session counts before and after update, candidate
IDs, and float64 scores.

## Temporal state and joins

One fixed-capacity scheduler-owned state stores:

- conversation generation;
- whether a previous decode layer-24 event exists;
- previous forward/decode index and selected IDs;
- 128 fixed session counters;
- pending top-16 prediction and score vector;
- predictor completion timestamp;
- source layer-24 observation timestamp;
- artifact and configuration identity;
- fixed-capacity completed records.

The first decode layer-24 event seeds state and creates a prediction for the
next token, but produces no completed transition record. Every later decode
event must have the next forward index and decode sequence index. It first
joins the pending prediction to the authoritative current IDs, records timing
and ranking metrics, then updates session counts and creates the next pending
prediction.

The path rejects prefill, duplicate/skipped identities, invalid IDs, stale
generations, non-finite scores, overflow, and incomplete teardown. A dedicated
reset API increments the conversation generation, clears counters and pending
state, and is called explicitly by the canonical probe before inference.

## Timing boundary

The canonical host callback already has the authoritative layer-24 IDs. T1
records `steady_clock` timestamps without requesting tensors or synchronizing:

- source observation time for token `t`;
- predictor completion / earliest hypothetical queue time;
- target layer-24 observation time for token `t+1`.

Available lead time is target observation minus earliest queue time. This is a
host-wall scheduling window, not proof of CUDA overlap.

The analysis combines this live distribution with separately labeled
reference costs:

- measured P2 packed expert H2D CUDA event: approximately `0.306016 ms`;
- measured P2 host staging time: `1.0254 ms`;
- measured P2 host enqueue time: `0.0222 ms`;
- a predeclared conservative safety margin of `0.250 ms`.

It reports minimum, p5, p50, p95, maximum, domains/prompts, and counts meeting
H2D-only, staging+H2D, and full conservative deadlines.

## One-transfer shadow rule

T1 reconstructs a separate conversation-reset 32-slot reactive LRU in analysis
only. For each transition it selects the highest-ranked supported candidate
that is not resident and has a safely reservable LRU slot. No runtime cache
state is read or changed.

Report hit@1/2/4, recall@8/12/16, highest true rank, MRR, candidates already
resident, chosen missing candidate/rank, useful/wasted/no-admission counts,
estimated ready-useful and late-useful counts, possible avoided misses, and
eviction regret. These measurements cannot retune T0.

## Validation

Use the fixed general, code, and translation prompts with one warmup and at
least three measured repetitions per disabled/enabled mode. Require exact
prompt/generated tokens and ordered router events, deterministic live
predictions, offline/live golden equivalence, correct reset, stable VRAM and
host memory, no cache mutation, and complete cleanup.

T1 passes only if equivalence and exactness hold, overhead is negligible, and
the measured lead-time plus one-transfer usefulness leave a credible bounded
T2 path. T1 never performs a transfer and does not require or claim speedup.

