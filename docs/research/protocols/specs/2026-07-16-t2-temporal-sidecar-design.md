# T2 temporal two-slot sidecar design

## Status

Approved bounded implementation design. T1 is preserved at ExpertFlow commit
`32b13ac` and llama.cpp commit `b9445cd5`. Both commits remain unmerged and
unpushed.

T2 is disabled by default. `live_cache_enabled=false` remains the release
default. The true layer-24 router remains authoritative.

## Objective

Determine whether one frozen temporal prediction per decode token can become a
ready, useful layer-24 packed-Q4 expert and reduce measured reactive blocking
without allowing speculative work to evict or reorder the proven 32-slot
reactive LRU.

This is one layer, one predicted transfer, two dedicated sidecar slots, and one
unchanged CUDA expert operation. It is not permission for confidence gating,
multiple predicted transfers, another layer, retraining, MTP, or graph/backend
placement work.

## Alternatives considered

1. Admit predictions into the normal 32-slot LRU. Rejected because T1 projected
   seven speculative eviction-regret cases.
2. Allocate a separate two-expert tensor and combine it with the normal arena.
   Rejected because the current `MUL_MAT_ID` operation requires one compatible
   packed tensor and one physical-ID space.
3. Extend the existing layer-24 packed tensor to 34 physical slices while
   retaining a separate 32-entry reactive policy state. **Selected.** It keeps
   execution on the proven operation and makes speculative ownership explicit.

## Fixed architecture

Layer 24 owns one contiguous packed arena with 34 physical slots:

- physical slots 0–31: normal reactive cache;
- physical slot 32: `SPECULATIVE_A`;
- physical slot 33: `SPECULATIVE_B`.

The existing reactive cache state, LRU sequence, generation counters, hit/miss
accounting, and replacement planner remain exactly 32 slots. The sidecar owns a
separate two-entry state machine. Reactive planning cannot inspect, allocate,
evict, or commit slots 32–33.

The scheduler-generated gate/up, down, and scale tensors expose 34 slices only
when the explicit T2 sidecar feature is enabled for layer 24. The unchanged
`GGML_OP_MUL_MAT_ID` consumes the same three packed tensors and remapped physical
IDs. No CUDA kernel, GGML operation, graph structure, allocator architecture,
or tensor representation changes.

The known packed payload is 3,345,412 bytes per expert. Two sidecar experts add
6,690,824 packed bytes. The exact backend allocation, component offsets,
alignment padding, and measured VRAM delta must be captured from the built
runtime; projection is not reported as allocation.

## Configuration

Use a new explicit environment feature:

`EXPERTFLOW_T2_TEMPORAL_SIDECAR=1`

It is valid only when all of the following hold:

- temporal shadow is enabled with the frozen T1 artifact;
- live cache is explicitly enabled in blocking mode;
- exactly layer 24 is cache-enabled;
- the layer-24 MoE path is CUDA-resident;
- pipeline copy count is one;
- the CUDA prefetch service resolves successfully;
- the arena exposes exactly 34 layer-24 physical slices;
- a T2 deferred log path and valid run ID are supplied.

Absent the variable, T1 and C5 remain byte-for-byte on their existing paths.
Invalid combinations abort before inference or arena allocation.

## Sidecar state

Each sidecar entry stores:

- physical slot ID, fixed to 32 or 33;
- CUDA descriptor ID, fixed to 0 or 1;
- logical expert ID;
- slot generation;
- transfer generation;
- source decode index;
- target decode index;
- conversation generation;
- state;
- transfer metrics;
- whether the true router demanded the expert;
- whether execution used the sidecar copy.

States:

- `EMPTY`
- `RESERVED`
- `STAGING`
- `TRANSFER_IN_FLIGHT`
- `READY`
- `IN_USE`
- `EXPIRED`

Legal lifecycle:

`EMPTY/EXPIRED -> RESERVED -> STAGING -> TRANSFER_IN_FLIGHT`

Then exactly one of:

- `TRANSFER_IN_FLIGHT -> READY -> IN_USE -> EXPIRED`
- `TRANSFER_IN_FLIGHT -> EXPIRED` after a completed unused transfer
- `TRANSFER_IN_FLIGHT -> READY -> EXPIRED` when ready but unused

An `EXPIRED` slot returns to `EMPTY` only after its matching CUDA descriptor is
confirmed inactive. Illegal transitions, stale token identities, stale
conversation or transfer generations, expert mismatches, descriptor mismatches,
or reuse before completion fail closed.

Target decode index parity selects the sidecar deterministically:

- even target decode index: slot 32 / descriptor 0;
- odd target decode index: slot 33 / descriptor 1.

If the selected ping-pong slot is not safely reusable, T2 records
`NO_SAFE_SIDECAR` and issues no speculative transfer. It never waits for an
unused old transfer solely to admit new speculation.

## Prediction and enqueue

After the authoritative layer-24 observer receives decode token `t`:

1. Update the frozen T1 temporal state in the accepted order.
2. Produce the ordered width-16 candidates for token `t+1`.
3. Snapshot the current 32-slot reactive mapping without mutation.
4. Traverse candidates in order, skipping every expert resident in slots 0–31.
5. Select the first remaining supported candidate.
6. Select the target token's deterministic ping-pong slot.
7. Validate the slot/descriptor is safely reusable.
8. Reserve the next nonzero generation.
9. Build exact source and destination ranges for physical slot 32 or 33.
10. Stage and enqueue exactly one three-component transfer on the existing
    non-blocking CUDA stream.
11. Return without compute-stream synchronization.

The frozen predictor weights, support filter, candidate width, and tie-breaking
do not change.

## Demand reconciliation and mixed mapping

At token `t+1`, the existing `expertflow_cache_prepare_selected` boundary reads
the eight authoritative logical IDs before expert execution.

If an active sidecar record targets this exact decode and conversation:

- query its matching CUDA event;
- if demanded and ready, classify `READY_USEFUL`;
- if demanded and incomplete, classify `LATE_USEFUL`, safely wait for its exact
  generation, then use it;
- if not demanded, classify `WASTED`; query completion and expire it when safe.

At most one logical expert can be supplied by a sidecar in an invocation. A
sidecar-aware pure planner:

- maps that logical expert to physical slot 32 or 33;
- runs the existing ordered reactive planning rules for the other seven
  demands against the unchanged 32-entry cache state;
- does not count the sidecar expert as a reactive hit or miss;
- does not load, promote, or commit the sidecar expert into slots 0–31;
- updates normal LRU recency only for normal reactive mappings;
- preserves the eight original logical IDs, order, and routing weights.

After all reactive misses are blocking-loaded and the sidecar event is complete,
the eight physical IDs are written to the existing selected-ID tensor. The
unchanged `MUL_MAT_ID` executes directly from the single 34-slice packed tensor.

If the sidecar is absent or unusable, all eight demands use the unchanged C5
reactive fallback.

## CUDA transfer service

Reuse the existing opaque CUDA prefetch service with:

- one non-blocking stream;
- two fixed descriptors;
- two fixed pinned staging regions;
- per-descriptor start/end events;
- no per-token allocation;
- descriptor 0 bound to slot 32;
- descriptor 1 bound to slot 33.

Query and wait use exact descriptor and transfer generations. A descriptor
becomes reusable only after query/wait reports the matching generation ready
and clears its active state.

No new CUDA kernel or operation is permitted.

## Telemetry

Use fixed-capacity preallocated records and deferred output. Per predicted
transition record:

- source and target decode indices;
- conversation generation;
- predicted expert and rank;
- physical sidecar slot and descriptor;
- slot and transfer generations;
- state transitions;
- true eight-expert demand;
- `READY_USEFUL`, `LATE_USEFUL`, `WASTED`, `NO_CANDIDATE`, or
  `NO_SAFE_SIDECAR`;
- whether execution used the speculative copy;
- staging, enqueue, CPU-return, queue-to-ready, and CUDA-event H2D timing;
- bytes transferred and wasted;
- blocking wait caused by a late useful transfer;
- reactive hits, misses, bytes, and blocking time separately;
- final 32-slot reactive mapping;
- exact arena allocation and component offsets.

Aggregate:

- ready-useful, late-useful, wasted, no-candidate, and no-safe-slot counts;
- normal reactive hit rate;
- sidecar ready-hit rate;
- blocking misses and blocking milliseconds per generated token;
- reactive blocking reduction;
- H2D/staging/queue distributions;
- bytes and wasted bytes;
- prompt TPS, decode TPS, end-to-end time, TTFT, and available p50/p95 token
  latency;
- peak VRAM, process memory, exact arena bytes, and cleanup.

Host-wall timestamps, CUDA-event duration, and inferred overlap remain clearly
separated.

## Test-first ladder

### S0 - pure sidecar state

Test legal lifecycle, illegal transitions, generations, ping-pong selection,
expiry, reuse, stale tokens, and descriptor ownership.

### S1 - 34-slot layout

Test that ordinary cache mode remains exactly 32 slices and T2 layer 24 becomes
exactly 34. Verify component offsets, bounds, packed payload, aligned allocation,
and physical slots 32/33. Test that reactive copy ranges still reject slots
above 31 while sidecar ranges accept only 32/33.

### S2 - mixed planner

Test one external sidecar mapping plus seven reactive mappings. Require:

- original logical order;
- physical sidecar ID only for the declared expert;
- no sidecar expert in normal loads or commits;
- identical reactive decisions for remaining demands;
- no normal state mutation during planning;
- normal commit validation excluding the external slot;
- rejection of duplicate, stale, or multiple external mappings.

### S3 - compiled but disabled

Build the 34-slot-capable source with T2 disabled and reproduce T1/C5 behavior,
tokens, routing, 32-slot allocation, and cache events exactly.

### S4 - enabled infrastructure, no transfer

Enable T2 with no eligible candidate or deliberately unavailable sidecar.
Require exact reactive parity, no arena overrun, and exact 34-slot allocation.

### S5 - controlled ready-useful

Transfer one known demanded expert to the selected sidecar before demand. Execute
the unchanged operation with mixed reactive/sidecar IDs and require exact
tokens, routing, byte ranges, and protected LRU state.

### S6 - controlled late-useful and wasted

Force one incomplete demanded transfer and one unused transfer. Verify safe wait
only for demanded work, expiry without normal-cache mutation, descriptor reuse,
and exact execution.

### S7 - unconditional temporal sidecar

Run the frozen predictor on the focused general, code, and translation suite,
one warmup plus three measured repetitions per mode:

1. reactive C5, 32 slots;
2. temporal sidecar, 32 reactive plus two speculative slots.

## Passing criteria

T2 passes only when:

- at least one useful prediction is ready before demand;
- prompt/generated tokens and all recorded router data are exact;
- normal reactive slots experience zero speculative eviction;
- the mixed mapping executes through unchanged `MUL_MAT_ID`;
- reactive blocking time decreases;
- combined predictor, staging, transfer, waiting, remapping, and bookkeeping
  overhead does not exceed saved blocking time;
- memory and CUDA resources remain stable;
- feature-off restores the existing path.

Waste rate alone is not failure. Wasted-copy overhead is judged by measured
blocking and end-to-end performance.

If unconditional T2 is exact but measurably slower, stop and report. Confidence
gating requires a new authorization and validation-only frozen threshold.

## Immediate stop conditions

Stop without workaround if mixed execution requires:

- separate incompatible expert tensors;
- a CUDA kernel change;
- GGML graph restructuring;
- a new execution operation;
- repacking experts;
- general allocator redesign;
- unsafe slot aliasing or synchronization;
- speculative mutation of slots 0–31;
- more than one predicted transfer per token;
- per-token GPU or pinned-host allocation.

Also stop on parity drift, stale execution, memory growth, CUDA-resource leaks,
or inability to keep T2 disabled by default.
