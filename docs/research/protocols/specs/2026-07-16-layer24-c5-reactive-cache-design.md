# Layer-24 C5 Reactive Cache Design

**Status:** C5-0 through C5-4 passed on 2026-07-16; default remains disabled.

## Contract

C5 expands the verified C4 layer-24 primitive from eight to exactly 32 persistent packed CUDA expert slices. The unchanged CUDA `MUL_MAT_ID` operations consume that 32-slice operand directly; authoritative top-8 logical IDs are remapped to physical IDs 0–31 while selected order and routing weights remain unchanged. There is no eight-slice staging operand or demand-side staging copy.

The cache remains disabled by default and isolated to layer 24. It uses true-router demand, blocking host-to-CUDA copies, direct Q4_0/F32 slices without repacking, one scheduler-owned allocation, and the existing bounded event infrastructure. It adds no prediction, asynchronous stream, MTP, ML, second layer, CUDA kernel change, model-graph rewrite, or general allocator change.

## Residency and eviction

State resets at conversation/runtime start. Every invocation protects all currently demanded resident experts. Existing demanded experts are hits and retain their slots. Missing experts are processed in authoritative selected order; each takes the lowest free slot, otherwise the unprotected least-recently-used slot, with ascending slot ID breaking equal-recency ties. Access recency is committed only after a complete validated blocking transfer set, so a failed invocation cannot publish partial mappings. Forced-miss testing follows the same deterministic replacement order.

All 32 slots carry logical expert ID, generation, and last-use sequence. Before execution, validate logical ID, exact packed component sizes, tensor type/shape/strides, source/destination bounds, generation, completed transfer, and the full top-8 mapping. Fail explicitly on duplicates, invalid IDs, incomplete transfers, stale mappings, unsafe eviction, overflow, or metadata mismatch.

The projected aligned contract is `32 * 3,346,048 = 107,073,536` bytes (102.113 MiB). The report must distinguish this conservative projection from the backend's exact measured allocation.

## Validation ladder

- C5-0: compiled and disabled; exact C4/canonical parity.
- C5-1: enabled passthrough; infrastructure only, exact parity and stable memory.
- C5-2: known-set fill and repeat hits; exact packed bytes, mappings, tokens, routing, and stable execution.
- C5-3: deterministic replacement and forced misses; exact LRU/tie-break generations, no stale reads, and exact parity.
- C5-4: arbitrary true-router layer-24 sets over the focused deterministic suite, repeated; exact prompt/generated tokens, selected IDs, order/weights where recorded, event ordering, determinism, memory stability, and cleanup.

At the passing milestone, run the focused C++ tests, all 89 ExpertFlow tests, judge replay, feature-off restoration, and repeated process/memory checks. Event logs remain concise and include selected logical IDs, physical IDs, hits/misses, retained/replaced experts, bytes, blocking duration, generation changes, recency, and final residents.

## Stop conditions

Stop without expansion if direct packed execution requires repacking, an eight-slice staging copy, a CUDA kernel/graph/general-allocator redesign, stale or unsafe slot use, parity drift, persistent allocation growth, or loss of feature isolation/disabled-default behavior. Passing C5 does not authorize prediction, asynchronous overlap, another layer, or runtime-speed claims.
