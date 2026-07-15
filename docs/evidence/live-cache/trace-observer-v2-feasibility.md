# Trace Observer v2: Non-Segmenting Boundary Feasibility

**Date:** 2026-07-15

**llama.cpp revision inspected:** `a7312ae94f801fc9c6786dc56e38df57b964f697`

**Verdict:** `CONDITIONAL-GO-FOR-ONE-LAYER-OBSERVER-PROTOTYPE`

**Implementation status:** not started

**Gate 4:** closed

**Default runtime:** `live_cache_enabled=false`

## Scope and preserved floor

This investigation starts from local merge commit `62fee62bd54775a13257ddc4b52aa40e7309a701`, which retains observer-isolation commit `aec5dd1ae0171f9814a4124827a2c73ba5089aff`. The protected Observatory, the pinned llama.cpp source, the isolation artifacts, and the withdrawn historical `93.28%` claim remain unchanged. All callback-derived real-model traces remain quarantined and corpus collection remains stopped.

The purpose of this note is only to identify a boundary that could observe one layer's authoritative selected expert IDs without selecting the callback scheduler path. It does not authorize or claim a repaired observer.

## Source facts

1. `src/llama-graph.cpp:1915-1918` builds the authoritative top-k as an I32 view over `GGML_OP_ARGSORT` and names it `ffn_moe_topk-{layer}`.
2. `src/llama-graph.cpp:1975-2010` and `2098-2099` pass that tensor as `src[2]` to the gate/up/down `GGML_OP_MUL_MAT_ID` consumers.
3. `src/llama-context.cpp:2436-2442` assigns those stable tensor names during normal graph construction. This naming does not require a user evaluation callback.
4. `ggml/src/ggml-backend.cpp:1576-1660` contains an existing selected-expert offload optimization. For host-resident MoE weights feeding `MUL_MAT_ID`, it obtains `node->src[2]`, copies the IDs to a host vector, synchronizes the IDs backend, builds the used-expert bitset, and transfers only selected expert ranges.
5. The host ID materialization at `ggml/src/ggml-backend.cpp:1604-1607` already occurs on this path. Recording the target layer immediately after it requires no additional tensor request, graph view, backend synchronization, or allocator lifetime extension.
6. `ggml-backend.cpp:1677-1708` proves the unsafe callback path is structurally different: registering a callback replaces whole-split asynchronous execution with node-range graph views and synchronizes each range. Observer v2 must leave `callback_eval` unset.
7. `ggml/src/ggml-cuda/argsort.cu:251-290` produces the full sorted I32 indices on the active CUDA stream. Instrumenting this producer is possible, but it would require new CUDA telemetry storage and drain plumbing even though the selected-expert offload path already has the same authoritative IDs on host.
8. `ggml/src/ggml-alloc.c:841-847` and `1068-1075` reset and reuse graph allocation buffers. Retaining a selected-ID tensor pointer after graph execution is therefore not a safe extraction contract without pinning or copying its data before reuse.

## Ordered boundary assessment

### 1. Backend operation consuming selected IDs — conditional go

The smallest viable boundary is the existing `GGML_OP_MUL_MAT_ID` host-weight/offload branch in `ggml/src/ggml-backend.cpp`, immediately after lines 1604-1607 have materialized and synchronized `ids`.

For the first proof, the observer would:

- remain disabled unless explicit observer-only environment settings provide one layer ID and one output path;
- match `ids_tensor->name` exactly to `ffn_moe_topk-{layer}`;
- require I32, top-k 8, and one-token microbatches; fail closed on any mismatch;
- record the eight IDs once per distinct selected-ID tensor inside the existing `ids_tensor != prev_ids_tensor` branch, avoiding duplicate gate/up/down records;
- use a scheduler-owned, bounded observer state and append-only output;
- never register `callback_eval`, request a tensor, add a graph node/view, change selected IDs, or alter expert transfers;
- emit zero records and perform no allocation/file access when disabled.

This boundary is conditional because it is reached only when MoE weights are host resident and copied to a backend for `MUL_MAT_ID`. The prototype must first prove the selected target layer reaches this exact branch under the frozen ten-layer configuration. If it does not, the experiment stops; it must not broaden into graph partitioning or an allocator/backend rewrite.

### 2. Deferred post-graph extraction — no-go

Keeping `ffn_moe_topk-{layer}` and reading it after the graph is not independently safe. It is an intermediate view backed by scheduler/gallocr storage that is reset and reused. Making it persist would require an in-graph copy, a requested callback tensor, allocator pinning, or a new public lifetime contract. The first two recreate the perturbing/segmenting behavior; the latter two exceed the bounded scope.

### 3. Device-side ring buffer — defer

A ring buffer launched after CUDA `GGML_OP_ARGSORT` could preserve the normal scheduler path, but it requires CUDA context state, device/pinned allocation lifetime, layer metadata, capture-safe kernel/copy behavior, overflow handling, and a host drain API. It is materially larger than using the existing host ID materialization. It should be reconsidered only if the one-layer offload boundary is absent or parity-safe but insufficient, and only under a new written gate.

## Intended prototype change

The proposed source change is limited to a separate llama.cpp linked worktree at the pinned revision:

- modify `ggml/src/ggml-backend.cpp` only;
- add approximately 90-140 lines: a private POD observer configuration/state, strict environment parsing during scheduler creation, one bounded capture block after the existing ID read, and close/flush during scheduler destruction;
- use the existing `ids` vector and existing synchronization; add no CUDA source, graph-builder, allocator, public API, model-format, or cache-manager change;
- keep the patch behind an observer-specific flag, separate from and never implying `live_cache_enabled`.

No source modification is made by this note.

## Required prototype tests

The future prototype is acceptable only if all of the following pass against the unchanged clean CUDA baseline:

1. With observer variables unset, binary behavior, tokens, routing path, logs, and allocations match the baseline and no observer file is created.
2. With one layer enabled, the exact target branch is encountered once per forward and emits eight in-range IDs in causal order.
3. General, code, and translation trace-off versus observer-on prompt/generated tokens are exactly equal for at least three repetitions each.
4. Observer-on repetitions emit identical ordered IDs; strict trace validation passes after conversion to the canonical schema.
5. Forced one-token microbatches preserve the frozen invocation; any larger batch fails closed rather than inventing token coordinates.
6. GPU/host memory settles after repeated runs with no persistent process or allocation growth.
7. All ExpertFlow tests and judge replay pass; protected Observatory and pinned clean llama.cpp remain clean.
8. Disabling the observer restores the original path and output byte-for-byte at the source-control diff level.

## Stop conditions

Stop without implementation expansion if the target layer does not traverse the existing host-weight `MUL_MAT_ID` branch, if a new synchronization is needed, if absolute token mapping cannot be made correct under one-token microbatches, if exact token parity fails once, if records duplicate or reorder, or if the change needs `llama-context.cpp`, `llama-graph.cpp`, CUDA kernels, callback tensor requests, graph partitioning, allocator pinning, or a broad backend rewrite.

## Recommendation

Proceed only to a separately approved, one-file, one-layer observer prototype at the existing `ggml-backend.cpp:1604-1607` host-ID materialization point. This is the only investigated option that reuses an already-required authoritative ID read and synchronization while preserving the normal whole-split scheduler path. Gate 4 remains closed until that prototype proves exact multi-domain token parity and stable, complete IDs. No cache implementation, replacement corpus collection, runtime speedup claim, or restored policy claim is authorized by this recommendation.
