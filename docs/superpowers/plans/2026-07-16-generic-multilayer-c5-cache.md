# Generic Multi-Layer C5 Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the exact layer-24 C5 cache into a disabled-by-default, consolidated 32-slot-per-layer cache and validate it at two, five, then all 30 Gemma MoE layers.

**Architecture:** Parse exact `blk.N` component names into fixed per-layer contexts, keep independent C5 LRU state per enabled layer, and back every layer's three direct packed tensor views with one aligned CUDA allocation. Advance through explicit layer sets only after cache-off/cache-on exactness, memory, blocking, and throughput gates pass.

**Tech Stack:** C++17, MSVC 19.39/v143, CUDA 12.8, CMake/Ninja, pinned llama.cpp C5 lineage `641f5313`, Python 3.12/pytest, NVIDIA SMI sampling.

## Global Constraints

- Begin llama.cpp source work only after diagnostic benchmark commit `c72f578`.
- Modify the isolated multi-layer llama.cpp and ExpertFlow worktrees only; preserve the protected Observatory and C5 milestones.
- Exactly 32 direct packed slots per enabled layer; initial ramp sets are `[0,24]`, `[0,7,14,21,29]`, then `0..29`.
- Default disabled; true-router reactive demand and blocking copies only.
- No prediction integration, async copy stream, MTP/ML, 64-slot test, repacking, kernel change, graph rewrite, or per-token allocation.
- Use RED-GREEN-REFACTOR, commit each passing ramp separately, and append every command, result, failure, measurement, artifact, and decision.
- Treat host-wall blocking, CUDA-event latency, GPU sampling, exact backend allocation, and simulated results as distinct measurements.

---

### Task 1: Generic layer metadata and configuration

**Files:**
- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`
- Modify: `native/router_probe/live_cache_config.h`
- Modify: `native/router_probe/live_cache_config.cpp`
- Modify: `native/router_probe/test_live_cache_config.cpp`
- Modify: `native/router_probe/main.cpp`

**Interfaces:**
- Consumes: exact Gemma component names and environment settings.
- Produces: `expertflow_cache_tensor_identity { int layer_id; expertflow_cache_component component; }`, `expertflow_cache_layer_layout`, and an ascending explicit layer list.

- [ ] **Step 1: Add RED tensor-identity tests**

  Test exact matches for layers 0, 24, and 29; return both layer and component; reject layer 30, negative/non-decimal IDs, partial names, suffixes, wrong Q4_0/F32 types, shapes, strides, or 128-expert count. Replace `expertflow_cache_layer_24_layout()` with:

  ```cpp
  bool expertflow_cache_identify_tensor(
      const char * name,
      int type,
      const std::array<int64_t, 4> & ne,
      const std::array<size_t, 4> & nb,
      expertflow_cache_tensor_identity & identity);

  bool expertflow_cache_get_layer_layout(
      int layer_id,
      expertflow_cache_layer_layout & layout);
  ```

- [ ] **Step 2: Run the native cache test and confirm RED**

  Run the existing assertion-active `test-expertflow-cache` target. Expected: compile failure for the missing generic types/functions, followed by behavioral failures until exact-name validation exists.

- [ ] **Step 3: Implement strict generic identification**

  Parse only `blk.<decimal>.<exact component suffix>`, require `0 <= layer_id < 30`, and apply the existing layer-24 metadata contract to every layer. Keep planner constants at eight demands, 32 slots, and 128 experts.

- [ ] **Step 4: Add RED configuration tests**

  Require:

  ```text
  EXPERTFLOW_LIVE_CACHE=1
  EXPERTFLOW_LIVE_CACHE_MODE=blocking
  EXPERTFLOW_LIVE_CACHE_LAYERS=0,24
  EXPERTFLOW_LIVE_CACHE_LOG=C:\...\cache.jsonl
  ```

  Test exact parsing of `[0,24]`, `[0,7,14,21,29]`, and `0..29`; reject empty/unsorted/duplicate/out-of-range/non-decimal/whitespace/range syntax. Preserve singular `EXPERTFLOW_LIVE_CACHE_LAYER=24`; reject singular plus plural together. Verify absent master flag remains disabled and subordinate settings alone fail.

- [ ] **Step 5: Implement the explicit layer-list contract**

  Use a fixed `std::array<bool, 30>` plus fixed ordered IDs/count; do not heap-allocate during parsing. Generate exactly three anchored tensor override patterns per enabled layer. Keep passthrough and blocking validation behavior unchanged.

- [ ] **Step 6: Run focused tests GREEN and commit**

  Run both native configuration/cache tests and the existing C5 regressions. Commit the generic metadata/configuration milestone only after disabled and legacy layer-24 modes pass unchanged.

### Task 2: Fixed per-layer contexts and consolidated layout planner

**Files:**
- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`

**Interfaces:**
- Consumes: Task 1's ordered enabled layer IDs and per-layer component allocation sizes/alignment.
- Produces:

  ```cpp
  struct expertflow_cache_layer_context {
      bool enabled;
      expertflow_cache_state state;
      uint64_t event_count;
      uint64_t demand_count;
      uint64_t hit_count;
      uint64_t miss_count;
      uint64_t bytes_transferred;
      uint64_t blocking_duration_us;
  };

  struct expertflow_cache_arena_layout {
      size_t total_bytes;
      std::array<expertflow_cache_layer_arena_layout, 30> layers;
  };

  bool expertflow_cache_plan_arena(
      const expertflow_cache_runtime_config & config,
      size_t alignment,
      const std::array<expertflow_cache_component_sizes, 30> & sizes,
      expertflow_cache_arena_layout & layout);
  ```

- [ ] **Step 1: Add RED isolation and offset tests**

  Prove that hits, generations, recency, reset, and forced misses in layer 0 do not alter layer 24. Test ascending layer placement, per-component padding, no overlaps, exact bounds, overflow rejection, and projected packed payloads of `214,106,368`, `535,265,920`, and `3,211,595,520` bytes before backend alignment for 2/5/30 layers.

- [ ] **Step 2: Run the focused test and confirm RED**

  Expected: missing context/layout types or singleton-state cross-layer failure.

- [ ] **Step 3: Implement fixed contexts and pure layout planning**

  Reuse the proven C5 planner unchanged inside each layer context. Compute `gate_up`, `down`, and `scale` offsets for every enabled layer in configured ascending order with checked size arithmetic and backend alignment.

- [ ] **Step 4: Run focused tests GREEN and commit**

  Include one-layer layout regression and exact no-overlap assertions. Commit before scheduler integration.

### Task 3: Consolidated scheduler allocation and direct execution

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`
- Create: `tests/test_multilayer_cache_source.py`

**Interfaces:**
- Consumes: Task 2's fixed contexts/layout and Task 1's tensor identities.
- Produces: one scheduler-owned CUDA buffer, per-layer tensor bindings/views, and exact direct execution for configured layers.

- [ ] **Step 1: Add RED scheduler-contract tests**

  Add source-contract tests that require arrays indexed by layer and component, one consolidated `ggml_backend_buft_alloc_buffer` call, a parsed layer ID on every binding, and no remaining hard-coded JSON `"layer_id":24`. Add native unit tests for rejecting duplicate/missing/cross-layer component bindings.

- [ ] **Step 2: Run focused tests and confirm RED**

  Expected: source-contract failures from singleton bindings/state/arena and hard-coded layer 24.

- [ ] **Step 3: Replace singleton scheduler members**

  Store fixed `[30]` layer contexts and `[30][3]` bindings, one arena buffer/context, exact allocation bytes, and bounded records. Keep `n_copies == 1`. Register a component only in its identified layer and reject conflicting roots or destinations.

- [ ] **Step 4: Allocate and bind one consolidated arena**

  Wait until every enabled layer has all three source/destination bindings. Create three tensor views per enabled layer, plan aligned offsets, allocate once, bind each view into its assigned subregion, and redirect only matching scheduler-generated inputs in the same layer.

- [ ] **Step 5: Generalize selected-ID preparation**

  Change the preparation entry point to:

  ```cpp
  static bool expertflow_cache_prepare_selected(
      ggml_backend_sched_t sched,
      int layer_id,
      ggml_backend_sched_split * split,
      int input_id,
      ggml_backend_t split_backend,
      ggml_tensor * node);
  ```

  Read authoritative IDs, plan/copy/synchronize/validate/commit only that layer, and publish physical IDs only after all three component copies complete. Track prepared ID tensors by layer so one layer cannot suppress preparation in another.

- [ ] **Step 6: Generalize teardown and event schema**

  Emit schema `1.2.0` with `layer_id` and `layer_access_sequence`, followed by the existing selected IDs, slots, hits/misses, loads, bytes, host-wall blocking duration, generations, and resident mapping. Write only at teardown; fail on bounded-record overflow.

- [ ] **Step 7: Build and run C0/C1**

  Build the pinned CUDA DLL/probe. With compiled code disabled, require exact parity with diagnostic cache-off. With passthrough enabled for `[0,24]`, require exact parity, no arena allocation, no cache events, stable memory, and clean exit.

- [ ] **Step 8: Run native/ExpertFlow regressions and commit**

  Run both native tests, the complete ExpertFlow suite, judge replay, JSON/JSONL parsing, feature-off restoration, and `git diff --check`. Commit the generic runtime infrastructure before blocking execution evidence.

### Task 4: Two-layer `[0,24]` ramp

**Files:**
- Create: `scripts/run_multilayer_cache_ramp.py`
- Create: `src/expertflow/benchmark/multilayer.py`
- Create: `tests/test_multilayer_cache_benchmark.py`
- Create: `tests/test_multilayer_cache_runner.py`
- Create: `docs/evidence/live-cache/multilayer-cache-ramp.md`
- Create: `configs/canonical-multilayer-cache.json`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: the fixed diagnostic protocol and schema `1.2.0`.
- Produces: reconciled repetition-level performance, parity, cache, allocation, and reserve evidence.

- [ ] **Step 1: Add RED parser/reconciliation tests**

  Use a two-layer fixture to require exact per-layer event counts, eight demands/event, `hits + misses = demands`, bytes equal misses times exact packed bytes, ordered router/cache correspondence, arena offsets within allocation, and aggregate totals equal the sum of layers. Reject missing/extra/duplicate/out-of-order layers and inconsistent mappings.

- [ ] **Step 2: Add RED runner-contract tests**

  Require one warmup plus three measured cache-off/cache-on repetitions for general, code, and translation; `-ngl 10`, greedy 64-token protocol; 100 ms NVIDIA memory samples; process cleanup; absolute event paths; machine/runtime/model hashes; exact environment capture; and separate exact allocation versus system-wide peak fields.

- [ ] **Step 3: Implement the minimum benchmark harness**

  Reuse `scripts/run_performance_diagnostic.py` and `src/expertflow/benchmark/performance.py` parsing/statistics. Add only multi-layer event reconciliation, allocation/reserve accounting, and layer tables.

- [ ] **Step 4: Run the two-layer matrix**

  Use `EXPERTFLOW_LIVE_CACHE_LAYERS=0,24`. Capture prompt/decode TPS, E2E, TTFT, token p50/p95 where available, counts, exact tokens/router events, actual arena, peak/settled GPU memory, host memory, per-layer/global hits/misses/bytes/blocking, blocking/generated token, reserve, and KV/state allocation/headroom.

- [ ] **Step 5: Apply the two-layer gate**

  Stop on any parity/event/mapping failure, process residue, allocation growth, OOM, severe unexplained TPS regression, or insufficient reserve. Compare against observer/cache-off and committed C5, and explicitly state whether added layer coverage lowers aggregate blocking and improves end-to-end decode behavior.

- [ ] **Step 6: Verify and commit the two-layer milestone**

  Run native tests, full ExpertFlow tests, judge replay, fixture reconciliation, artifact hashing, JSON/JSONL validation, feature-off restoration, and clean-worktree checks. Append measured evidence and commit before changing the layer set.

### Task 5: Five-layer and all-layer ramps

**Files:**
- Modify: `scripts/run_multilayer_cache_ramp.py`
- Modify: `src/expertflow/benchmark/multilayer.py`
- Modify: `tests/test_multilayer_cache_benchmark.py`
- Modify: `docs/evidence/live-cache/multilayer-cache-ramp.md`
- Modify: `configs/canonical-multilayer-cache.json`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: the passing Task 4 runtime and frozen benchmark protocol.
- Produces: separate committed five-layer and, only if approved by measurements, all-layer milestones.

- [ ] **Step 1: Run the five-layer set `[0,7,14,21,29]`**

  Do not change runtime logic. Repeat the exact Task 4 matrix and reconciliation. Require five cache events per complete forward where those layers are enabled and compare layer-specific locality/blocking against the train+validation selection rationale.

- [ ] **Step 2: Apply the five-layer gate and commit**

  Require exact parity, stable cleanup, measured arena/peak, per-layer accounting, decode TPS, reserve, and KV/state headroom. Stop if broader coverage regresses materially or reserve becomes unsafe. Commit five-layer evidence separately.

- [ ] **Step 3: Preflight the all-layer allocation**

  With cache disabled, record settled/peak GPU memory and exact non-cache runtime allocations. Add the measured 30-layer arena requirement plus explicit KV/state and desktop reserve. Do not start the matrix if the sum exceeds available VRAM or leaves less headroom than the five-layer observed peak variance plus the measured KV/state requirement.

- [ ] **Step 4: Run all intended layers `0..29` only after preflight passes**

  Repeat the same fixed prompt/repetition matrix. Require 30 ordered cache events per complete forward, exact cache-off/cache-on tokens and routing, per-layer and aggregate accounting, stable allocation/cleanup, and no stale or cross-layer state.

- [ ] **Step 5: Make the performance verdict**

  Compare all-layer cache against matched observer/cache-off, C5, the best earlier ramp, and stock `-ngl 99`. Report prompt/decode TPS, E2E, TTFT, token latency, cache blocking, actual arena/peak VRAM, reserve, and KV/state headroom. Do not claim CUDA overlap or final speedup unless directly measured.

- [ ] **Step 6: Final verification and commit**

  Run the full native and ExpertFlow suites, judge replay, deterministic parity, repeated cleanup, feature-off restoration, artifact hashing, JSON/JSONL parsing, and `git diff --check`. Commit the all-layer result separately if every gate passes. Leave 64-slot, async, and predictor integration closed for a new decision.
