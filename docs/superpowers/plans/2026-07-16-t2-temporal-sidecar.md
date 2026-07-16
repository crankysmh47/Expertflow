# T2 Temporal Two-Slot Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure one exact asynchronous temporal prefetch per decode token using two dedicated layer-24 sidecar slots without allowing speculation to mutate the protected 32-slot reactive LRU.

**Architecture:** Keep reactive policy state at 32 slots while conditionally exposing one contiguous 34-slice packed tensor to the unchanged `MUL_MAT_ID`. Add a separate two-slot CPU state machine, reuse the existing two-descriptor CUDA prefetch service, and extend cache planning with at most one validated external physical mapping.

**Tech Stack:** C++17, llama.cpp/GGML scheduler, CUDA 12.8, MSVC v143, CMake/Ninja, Python 3.12, pytest, JSON/JSONL evidence.

## Global Constraints

- Exactly layer 24.
- Exactly 32 reactive slots plus physical sidecar slots 32 and 33.
- Exactly one predicted asynchronous transfer per decode token.
- Frozen temporal predictor weights 0.5/0.4/0.1 and candidate width 16.
- No prediction may evict, admit into, or alter the LRU order of slots 0–31.
- No CUDA kernel, GGML operation, graph, backend placement, quantization, or repacking change.
- No per-token GPU or pinned-host allocation.
- T2 is disabled by default and requires explicit configuration.
- Stop immediately on any design stop condition.

---

### Task 1: Pure sidecar state machine

**Files:**
- Create: `ggml/src/ggml-expertflow-sidecar.h`
- Create: `ggml/src/ggml-expertflow-sidecar.cpp`
- Create: `tests/test-expertflow-sidecar.cpp`
- Modify: `ggml/src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `expertflow_sidecar_state`, `expertflow_sidecar_reserve`, `expertflow_sidecar_mark_staging`, `expertflow_sidecar_mark_in_flight`, `expertflow_sidecar_reconcile`, `expertflow_sidecar_mark_in_use`, `expertflow_sidecar_expire`, and deterministic slot selection.

- [ ] Write assertion-active tests for reset, even/odd target mapping, legal ready/late/wasted lifecycles, stale conversation/token/generation rejection, overflow, and reuse only after completion.
- [ ] Run `test-expertflow-sidecar` and verify RED because the API does not exist.
- [ ] Implement only the fixed two-slot state and transitions.
- [ ] Run the test and verify GREEN.
- [ ] Run the existing cache and temporal native tests.

### Task 2: Conditional 34-slot packed layout

**Files:**
- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`

**Interfaces:**
- Produces: per-layer physical slot count in `expertflow_cache_runtime_config`; a copy-range function accepting an explicit validated physical slot count; arena planning for 32 or 34 physical slices.

- [ ] Add failing tests proving default layouts remain 32, only enabled layer 24 may request 34, projected packed bytes equal `3,345,412 * 34`, slots 32/33 are bounded, and slot 34 is rejected.
- [ ] Run the cache test and verify RED.
- [ ] Add the minimal explicit physical-slot-count layout path without changing `expertflow_cache_state`.
- [ ] Run the cache test and verify GREEN.
- [ ] Verify existing 32-slot arena tests remain unchanged.

### Task 3: Sidecar-aware reactive planner

**Files:**
- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`

**Interfaces:**
- Produces: `expertflow_cache_external_mapping` and `expertflow_cache_plan_selected_with_external`.
- Consumes: one optional logical expert mapped to physical slot 32 or 33.

- [ ] Add failing tests for one external demand plus seven normal demands, logical-order preservation, normal no-mutation planning, normal-only commit, no sidecar promotion, duplicate/multiple/stale external rejection, and fallback equivalence when no external mapping exists.
- [ ] Run the cache test and verify RED.
- [ ] Implement the smallest planner extension by excluding the one external rank from normal allocation and recency updates.
- [ ] Add external-aware committed-mapping validation.
- [ ] Run cache tests and verify GREEN.

### Task 4: Explicit T2 configuration and feature-off contract

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-expertflow-sidecar.cpp`
- Modify: `tests/test_t1_temporal_source_contract.py`
- Create: `tests/test_t2_sidecar_source_contract.py`

**Interfaces:**
- Consumes: `EXPERTFLOW_T2_TEMPORAL_SIDECAR`, T2 log path, and run ID.
- Produces: scheduler-owned sidecar state, two-descriptor CUDA context, and deferred records.

- [ ] Add failing source/native tests for disabled default, exact required configuration, layer-24-only enforcement, two descriptors, no P1 mode dependency, and no CUDA/kernel/graph source changes.
- [ ] Run focused tests and verify RED.
- [ ] Parse T2 configuration before tensor duplication and set layer 24 physical count to 34 only for valid T2.
- [ ] Allocate fixed sidecar records and resolve the existing CUDA service with descriptor count two.
- [ ] Run focused tests and build.
- [ ] Run feature-off T1/C5 smoke and verify the arena remains 32 slots.

### Task 5: Enqueue temporal predictions into ping-pong slots

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-expertflow-temporal.h`
- Modify: `ggml/src/ggml-expertflow-temporal.cpp`
- Modify: `tests/test-expertflow-temporal.cpp`
- Modify: `tests/test-expertflow-sidecar.cpp`

**Interfaces:**
- Consumes: frozen `pending_prediction`, current 32-slot reactive state, target decode index, source tensor bindings, and arena addresses.
- Produces: at most one queued sidecar transfer and one deferred event record.

- [ ] Add failing tests for highest-ranked normal-cache-missing selection, deterministic descriptor parity, no-candidate, unsafe-slot no-admission, and exactly one transfer budget.
- [ ] Run tests and verify RED.
- [ ] Expose a narrow immutable temporal-result handoff after successful observation.
- [ ] Build exact slot-32/33 component ranges and enqueue one request.
- [ ] Transition the sidecar through RESERVED, STAGING, and TRANSFER_IN_FLIGHT.
- [ ] Run native tests and build.

### Task 6: Reconcile demand and execute mixed mapping

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test-expertflow-sidecar.cpp`
- Modify: `tests/test-expertflow-cache.cpp`

**Interfaces:**
- Consumes: authoritative eight logical IDs and the exact target-token sidecar record.
- Produces: one optional external physical mapping plus the unchanged reactive plan for uncovered demands.

- [ ] Add failing tests for ready-useful, late-useful, wasted, stale target, stale generation, safe expiry, and reactive fallback.
- [ ] Verify RED.
- [ ] Query the matching descriptor before planning.
- [ ] Wait only when the sidecar expert is truly demanded and incomplete.
- [ ] Supply ready/awaited demanded experts through physical slot 32 or 33.
- [ ] Run normal blocking loads only for uncovered demands, commit normal state only, and write all eight physical IDs in authoritative order.
- [ ] Transition used sidecars through IN_USE to EXPIRED after execution is safe.
- [ ] Run native tests and build.

### Task 7: Controlled runtime ladder

**Files:**
- Modify: `native/router_probe/main.cpp` only if a narrow T2 reset/identity call is required
- Modify: `native/router_probe/import/llama.def` only if a new public reset API is strictly necessary
- Create: `scripts/run_t2_sidecar_controlled.py`

**Interfaces:**
- Produces: S3–S6 raw token, router, cache, sidecar, transfer, performance, and memory evidence.

- [ ] Build an isolated T2 runtime and probe; record hashes.
- [ ] Run S3 compiled-disabled parity.
- [ ] Run S4 enabled/no-transfer parity and measure exact 34-slot allocation.
- [ ] Run S5 controlled ready-useful mixed execution.
- [ ] Run S6 controlled late-useful and wasted expiry/reuse.
- [ ] Stop immediately on parity, operation, graph, repacking, aliasing, synchronization, or memory failure.

### Task 8: Unconditional focused benchmark

**Files:**
- Create: `scripts/run_t2_temporal_sidecar_suite.py`
- Create: `scripts/summarize_t2_temporal_sidecar.py`
- Create: `src/expertflow/predictor/temporal_sidecar_analysis.py`
- Create: `tests/test_temporal_sidecar_analysis.py`

**Interfaces:**
- Produces: identical reactive-C5 and sidecar-predictive results for general, code, and translation with one warmup plus three measured repetitions.

- [ ] Write failing parser/summary tests for exactness, normal hit rate, ready/late/waste separation, timing classes, blocking reduction, allocation, and cleanup.
- [ ] Verify RED, implement the minimum strict analysis, and verify GREEN.
- [ ] Run the focused benchmark.
- [ ] Require exact prompt/generated tokens, router order/weights, determinism, and protected normal-cache behavior.
- [ ] Compute ready-useful, late-useful, wasted, blocking, CUDA-event, staging, queue, byte, TPS, latency, VRAM, allocation, and cleanup metrics.

### Task 9: Final decision and commits

**Files:**
- Create: `docs/evidence/live-cache/t2-temporal-sidecar-result.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: all controlled and focused evidence.
- Produces: PASS or bounded STOP verdict without confidence gating or expansion.

- [ ] Apply the written pass and stop criteria without retuning.
- [ ] Run the complete ExpertFlow suite, native cache/temporal/sidecar tests, judge replay, build, `git diff --check`, artifact hashing, and process cleanup.
- [ ] If the bounded experiment completes, commit llama.cpp implementation and ExpertFlow evidence separately with `Assisted-by: Codex` trailers.
- [ ] Keep commits unmerged and unpushed.
- [ ] Do not begin confidence gating, wider concurrency, or multi-layer work.
