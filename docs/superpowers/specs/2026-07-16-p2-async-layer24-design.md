# P2 asynchronous layer-24 prefetch design

## Status

Design only. Do not implement until the P1 commits and evidence are accepted.
P2 remains disabled by default and keeps the true layer-24 router authoritative.

## Objective

Use the frozen P1 layer-23 prediction to begin bounded asynchronous movement of
candidate Q4 experts into the already-proven layer-24 32-slot cache. Measure
whether ready prefetches reduce exact blocking misses and improve end-to-end
decode throughput without changing tokens, routing, memory stability, or the
protected Observatory.

## Fixed scope

- one transition: layer 23 to layer 24
- one target cache: layer 24, 32 slots
- frozen B2 source-normalized, phase-separated, width-12 predictor
- true-router authority
- exact blocking fallback
- one dedicated CUDA copy stream and per-slot completion events
- no multi-layer prediction
- no MTP, MLP, retraining, higher `-ngl`, or 64-slot work

## State machine

After layer-23 IDs are available, score the frozen top-12 candidates. For each
candidate:

1. Treat an already-resident expert as a predicted hit.
2. Reserve only a slot that is free or safely replaceable.
3. Never evict a slot in use by the current layer invocation.
4. Enqueue the packed Q4 copy on the dedicated stream.
5. Associate the logical expert, slot generation, byte range, and CUDA event.

At the authoritative layer-24 router:

1. Preserve its eight IDs, order, and weights.
2. Accept a predicted slot only when its generation and completion event match.
3. Count ready and late prefetches separately.
4. For a late or absent expert, use the existing exact blocking load.
5. Execute only after all eight true experts are verified resident.

Predictions never suppress a true-router demand or change the selected expert.

## Admission and replacement

Start with free-slot-only prefetch to prove event and lifetime correctness.
Then test a bounded predictive admission rule that may replace only the oldest
non-reserved, non-in-use slot. Retain the current reactive LRU state as the
fallback authority. A prediction may reserve at most 12 slots and must release
unused reservations after the layer-24 decision.

## Required telemetry

Per transition record:

- phase and forward index
- source IDs and predicted candidates
- true layer-24 IDs
- resident predicted hits
- copies enqueued
- ready, late, and unused prefetches
- exact blocking misses
- bytes prefetched, used, and wasted
- enqueue-to-ready CUDA-event latency
- blocking fallback time
- slot and generation changes

Aggregate p50/p95 latency, hit rate, useful-byte rate, copy/compute overlap,
decode TPS, prompt TPS, end-to-end time, peak VRAM, and cleanup.

## Validation ladder

1. P2-0: compiled but disabled; exact P1 parity.
2. P2-1: stream/events initialized with no copies; exact parity and stable
   memory.
3. P2-2: one fixed expert copied into a free slot and consumed only after its
   event completes.
4. P2-3: width-12 live predictions with free-slot-only prefetch.
5. P2-4: bounded replacement plus exact blocking fallback.
6. P2-5: focused general/code/translation measurements, then the seven-task
   suite.

## Stop conditions

Stop before expansion if:

- any prompt token, generated token, router ID, order, or weight differs;
- a stale generation or incomplete event can be consumed;
- packed Q4 data requires repacking;
- the change requires graph relocation or allocator redesign;
- GPU allocation grows across runs;
- warm no-copy execution has material unexplained overhead;
- asynchronous copies do not reduce measured blocking time;
- wasted bytes or synchronization erase plausible throughput headroom.

P2 passes only if exactness and cleanup remain intact and measured end-to-end
throughput improves against the identical reactive configuration. Miss
reduction alone is insufficient.
