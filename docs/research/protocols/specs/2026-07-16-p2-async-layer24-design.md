# P2 asynchronous layer-24 prefetch design

## Status

Accepted bounded implementation design. P1 remains preserved at ExpertFlow
commit `b50ea6a` and llama.cpp commit `6e7bdffe`. Those commits already descend
from the passing C5 layer-24 32-slot cache milestones, so the P2 worktrees are
the required combined runtime without an additional merge.

P2 is disabled by default. The true layer-24 router remains authoritative.

## Objective

Use the frozen P1 layer-23 prediction to begin bounded asynchronous movement of
candidate packed Q4 experts into the proven layer-24 32-slot cache. Determine
whether ready prefetches reduce exact blocking misses and improve end-to-end
decode throughput without changing tokens, routing, memory stability, or the
protected Observatory.

## Fixed scope

- one transition: layer 23 to layer 24
- one target cache: layer 24, 32 slots
- frozen B2 source-normalized, phase-separated, width-12 predictor
- true-router authority and exact blocking fallback
- one dedicated non-blocking CUDA copy stream
- fixed-capacity transfer descriptors and completion/timing events
- preallocated pinned staging storage where the source is not already pinned
- no graph callback, extra tensor request, graph segmentation, or relocation
- no multi-layer prediction
- no MTP, MLP, retraining, higher `-ngl`, or 64-slot work

## Architecture boundary

The scheduler in `ggml-backend.cpp` continues to own logical cache policy,
layer-24 tensor bindings, slot mappings, and exact fallback. CUDA-specific
stream, event, pinned-memory, and copy operations live inside the CUDA backend.
They are exposed to the scheduler through narrow opaque function pointers
obtained from `ggml_backend_reg_get_proc_address`.

This keeps the common backend free of direct CUDA linkage and lets the feature
fail closed when the required CUDA extension is unavailable.

An expert transfer is one logical operation containing the existing packed Q4
gate/up, down, and scale byte ranges. All three copies target the already
allocated slot arena. A completion event recorded after the third copy is the
only condition that can make the logical expert ready.

## Configuration

P1 shadow-only behavior remains available with live caching disabled. P2
requires all of the following:

- predictor shadow enabled
- live cache blocking mode
- exactly layer 24 enabled
- exactly 32 slots
- an explicit P2 prefetch mode

Absent the explicit P2 mode, the accepted P1 and C5 paths are unchanged.
Unsupported combinations abort during initialization with a specific reason.

## Slot state machine

Each physical slot has a logical expert ID, generation, and one of:

- `EMPTY`
- `RESIDENT`
- `RESERVED`
- `STAGING`
- `TRANSFER_QUEUED`
- `TRANSFER_IN_FLIGHT`
- `READY`
- `IN_USE`

Allowed progress is monotonic for a generation:

`EMPTY/RESIDENT -> RESERVED -> STAGING -> TRANSFER_QUEUED ->
TRANSFER_IN_FLIGHT -> READY -> IN_USE -> RESIDENT`

Cancellation before queueing restores the prior state. Once a copy is queued,
the slot cannot be reassigned until the event has completed and the result has
been reconciled. A generation mismatch, illegal transition, stale mapping,
incomplete transfer, or ownership conflict aborts the P2 path.

## Admission and replacement

Prediction produces an immutable admission plan before changing cache state.
The plan records the candidate expert, score/rank, proposed slot, prior
resident expert, prior generation, next generation, exact component ranges,
and reason for admission or rejection.

The initial policy admits only into an empty safe slot. Replacement may be
enabled only after that path passes. A replacement victim must be the oldest
slot that is not selected by the current authoritative invocation, reserved,
staging, queued, in flight, ready-but-unreconciled, or in use.

P2.2 admits at most one predicted expert per layer-23 to layer-24 transition.
P2.3 may increase the bounded number only after P2.0 measurements show that
staging capacity, transfer latency, overlap window, and memory headroom make it
useful. It is not a broad candidate-width or concurrency search.

## Runtime reconciliation

After layer-23 IDs are available:

1. Run the frozen predictor without changing its scores or candidate order.
2. Build a no-mutation admission plan.
3. Reserve the chosen slot and generation.
4. Stage the exact source bytes into the fixed pinned buffer if required.
5. Queue the three H2D ranges on the dedicated stream.
6. Record the completion event after the final copy.
7. Return control without synchronizing the compute stream.

At the authoritative layer-24 router:

1. Preserve all eight true IDs, their order, and routing weights.
2. Reconcile any outstanding prediction against its slot generation and event.
3. Accept it as ready only when the matching completion event has succeeded.
4. Count demanded-but-incomplete predictions as late.
5. Count completed predictions absent from the true demand as unused/wasted.
6. Use the existing exact blocking path for every absent or late true expert.
7. Execute only after all eight true experts have verified resident mappings.
8. Preserve the reactive cache as final replacement and execution authority.

No prediction can suppress a true demand, change an expert ID, change routing
order or weights, or make an incomplete slot executable.

## P2.0 transfer proof

Before prediction can mutate residency, prove one genuine asynchronous transfer
of one known layer-24 expert:

- dedicated CUDA copy stream
- fixed descriptor and pinned staging buffer
- exact three-component packed Q4 copy
- no compute-stream synchronization at enqueue
- CUDA events around H2D work
- host timing around staging, enqueue, return, and eventual reconciliation
- verification that independent compute can run before the completion wait
- exact byte and destination-bound checks

Report separately:

- pageable-to-pinned staging time
- host queue/enqueue time
- H2D CUDA-event duration
- host wall time from queue to ready
- CPU return latency after enqueue
- observed overlap window with intervening compute

Do not label host wall timing as CUDA-event latency or as overlap.

## Telemetry

Use fixed-capacity preallocated records in the live path and deferred output.
For each transition record:

- phase, forward index, and decode step
- source IDs and predicted candidates
- true layer-24 IDs
- admission decision and reason
- slot ID and generation before/after
- state transitions
- predicted resident hits
- transfers staged and enqueued
- ready, late, and unused prefetches
- exact blocking misses
- bytes prefetched, used, and wasted
- staging and host enqueue time
- CUDA-event H2D duration
- queue-to-ready host wall time
- blocking fallback time
- final logical-to-physical mapping

Aggregate p50/p95 latency, hit rate, useful-byte rate, ready/late/waste rates,
blocking-time reduction, prompt TPS, decode TPS, end-to-end time, peak VRAM,
allocation stability, and cleanup.

## Validation ladder

### Baseline

Before source changes, reproduce:

- predictor disabled with cache disabled
- exact C5 layer-24 reactive cache
- P1 offline/live-shadow output equivalence
- focused general, code, and translation prompts
- current binary/source/configuration hashes and ancestry

### P2.0 - asynchronous transfer feasibility

Queue one known expert without redirecting execution. Prove genuine asynchronous
copy behavior, exact ranges, stable allocation, event correctness, and measured
overlap opportunity.

### P2.1 - admission planning only

Run the predictor and build admission plans without mutating slots or issuing
copies. Require exact parity and deterministic plans.

### P2.2 - one predicted transfer

Admit at most one candidate per transition. Reconcile ready, late, and unused
transfers; preserve exact blocking fallback and parity.

### P2.3 - bounded multiple transfers

Only if P2.0 and P2.2 measurements justify it, freeze one small concurrency
budget and repeat the exactness and performance suite.

### Final comparison

Benchmark identical configurations:

1. cache disabled, prediction disabled
2. C5 reactive LRU
3. P2 predictive asynchronous prefetch with exact fallback

Use the same prompts, generation settings, repetitions, cache size, layer,
offload, and instrumentation level. Report repetition-level values and basic
variance.

## Passing criteria

P2 passes only if:

- prompt and generated tokens are exact
- router IDs, order, and weights are exact
- deterministic repetitions remain exact
- no incomplete or stale slot is consumed
- allocation and cleanup are stable
- asynchronous transfers reduce measured reactive blocking cost
- cache bookkeeping and instrumentation do not hide a severe regression
- end-to-end decode throughput improves against identical C5 reactive mode

Miss reduction alone is insufficient.

## Stop conditions

Stop without expansion if:

- any prompt token, generated token, router ID, order, or weight differs
- a stale generation or incomplete event can be consumed
- packed Q4 data requires repacking
- the implementation requires graph relocation or allocator redesign
- a dedicated stream cannot consume the proven arena safely
- source lifetime or staging safety cannot be guaranteed
- GPU or pinned-host allocation grows across runs
- warm no-copy execution has material unexplained overhead
- asynchronous copies do not reduce measured blocking time
- wasted bytes, synchronization, or metadata cost erase plausible headroom

On a stop condition, preserve the evidence and keep the accepted P1 and C5
milestones as the runtime floor.
