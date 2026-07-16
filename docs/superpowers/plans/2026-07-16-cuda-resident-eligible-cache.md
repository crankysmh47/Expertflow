# CUDA-Resident Eligible Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict the existing exact 32-slot ExpertFlow cache to MoE layers whose normal execution path is already CUDA-resident, then validate layers 21 and 24 together at `-ngl 10`.

**Architecture:** Preserve requested layers separately from runtime-eligible layers. Let the scheduler perform its ordinary backend assignment first, classify each requested layer using the naturally placed same-layer MoE router and expert consumers, then redirect only eligible expert consumers to the existing cache arena. Explicit requests fail if any layer is rejected; auto mode selects all compatible CUDA-resident layers.

**Tech Stack:** C++17, MSVC v143, CUDA 12.8, pinned llama.cpp cache branch, Python 3.12, pytest, CMake/Ninja.

## Global Constraints

- Preserve blocker evidence commit `06bf8a0` and all protected milestones.
- Default remains disabled.
- No graph relocation, whole-layer offload, hybrid expert operation, backend splitting, prediction, async copy, MTP, or 64-slot work.
- Eligibility comes from actual scheduler/backend placement, not only from the numeric `-ngl` value.
- Explicit requests fail on an ineligible layer; only explicit auto mode may filter requested layers.
- Use independent 32-slot contexts and one consolidated allocation for eligible layers.
- Do not commit llama.cpp changes without explicit user approval under the repository instructions.

---

### Task 1: Configuration and eligibility state

**Files:**
- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`
- Modify: `native/router_probe/live_cache_config.h`
- Modify: `native/router_probe/live_cache_config.cpp`
- Modify: `native/router_probe/test_live_cache_config.cpp`

**Interfaces:**
- Consumes: explicit layer list or `EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE=1`.
- Produces: fixed requested, eligible, and rejected layer sets plus an explicit/auto selection mode.

- [ ] Add RED tests requiring explicit and auto modes to be mutually exclusive, disabled-by-default behavior to remain unchanged, and explicit mode to preserve the exact requested list.
- [ ] Run native configuration tests and confirm failure for the missing auto/eligibility contract.
- [ ] Add fixed-capacity request/eligibility fields without heap allocation in the runtime parser.
- [ ] Run focused tests GREEN.

### Task 2: Natural backend eligibility classification

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test_multilayer_cache_source.py`

**Interfaces:**
- Consumes: ordinary pass-1 backend assignments and exact layer identities.
- Produces: a one-time classification with requested, eligible, rejected, backend/reason, slot count, and planned arena bytes.

- [ ] Add RED source/native tests proving the unconditional `1.efc` pass-1 accelerator selection is removed.
- [ ] Add RED tests requiring classification before cache-specific tensor duplication and redirection.
- [ ] Classify a layer eligible only when its same-layer MoE router path and expert consumers naturally resolve to the same CUDA backend and its exact tensor contract is complete.
- [ ] In explicit mode abort with layer/backend/reason on any rejection. In auto mode enable only validated layers.
- [ ] Log requested, eligible, rejected, configured `-ngl`, 32 slots per layer, and planned arena bytes before first execution.
- [ ] Run source/native tests GREEN and build the CUDA target.

### Task 3: Two-layer `[21,24]` exactness ramp

**Files:**
- Modify: `configs/canonical-multilayer-cache.json`
- Modify: `scripts/run_multilayer_cache_ramp.py`
- Modify: `src/expertflow/benchmark/multilayer.py`
- Modify: `tests/test_multilayer_cache_benchmark.py`
- Modify: `tests/test_multilayer_cache_runner.py`
- Create: `docs/evidence/live-cache/cuda-resident-two-layer-ramp.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: validated explicit eligible layers 21 and 24 at `-ngl 10`.
- Produces: exact cache-off/cache-on parity, independent per-layer cache accounting, allocation, memory, cleanup, and focused performance evidence.

- [ ] Add RED harness tests for exact eligibility-report parsing and rejection of missing or filtered explicit layers.
- [ ] Run one warmup and three measured repetitions for general, code, and translation.
- [ ] Reconcile exact tokens, ordered router events, layer events, mappings, generations, hits, misses, bytes, blocking time, allocation, peak/settled memory, and process cleanup.
- [ ] Run all ExpertFlow tests, replay tests, native tests, feature-off restoration, and `git diff --check`.
- [ ] Stop and report on any exactness, ownership, memory, cleanup, or severe unexplained TPS failure.

### Task 4: All eligible `-ngl 10` layers and performance gate

**Files:**
- Modify the Task 3 harness, evidence, configuration, and project log only after Task 3 passes.

**Interfaces:**
- Consumes: auto eligibility discovery under the same model/runtime.
- Produces: discovered indices, exact allocation, per-layer/global cache metrics, reserve, and comparison against stock, matched stock, and observer/cache-off.

- [ ] Discover the actual eligible CUDA-resident MoE layers in auto mode.
- [ ] Preflight the exact 32-slot allocation and safety reserve.
- [ ] Run the approved focused exactness/performance protocol.
- [ ] Decide whether a bounded `-ngl 15`/`20` sweep is justified.

### Task 5: Higher-offload, slot budget, and predictor gates

- [ ] Test only the small approved higher-offload set after Task 4 is exact and performance-credible.
- [ ] Test 64 slots only at the selected practical offload region and only when measured reserve permits.
- [ ] Integrate frozen B2 only after the best reactive point is selected, preserving true-router authority and exact blocking fallback.
