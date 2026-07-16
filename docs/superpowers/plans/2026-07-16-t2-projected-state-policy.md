# T2 Projected-State Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the final exact layer-24 sidecar experiment using frozen temporal candidates filtered against the projected post-current-token reactive cache state.

**Architecture:** Add one pure cache-policy helper that plans and commits the current authoritative demands on a copied state, then selects the highest-ranked candidate absent from that projected state. Replace only T2's current-state candidate filter with this helper; reuse the complete 34-slot sidecar, CUDA transfer, reconciliation, fallback, telemetry, and benchmark paths.

**Tech Stack:** C++17, llama.cpp/GGML scheduler, CUDA 12.8, MSVC v143, CMake/Ninja, Python 3.12, pytest, JSON/JSONL evidence.

## Global Constraints

- Exactly layer 24.
- Exactly 32 reactive slots plus sidecar slots 32 and 33.
- Exactly one predicted asynchronous transfer per decode token.
- Frozen temporal predictor weights, candidate width, ordering, and artifact.
- No retraining, retuning, confidence gating, wider concurrency, or cache-size change.
- No CUDA kernel, GGML operation, graph, placement, quantization, or repacking change.
- Live reactive state and LRU behavior remain authoritative and unchanged.
- T2 remains disabled by default.
- Freeze runtime development after the experiment regardless of outcome.

---

### Task 1: Pure projected-state selector

**Files:**
- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`

**Interfaces:**
- Consumes: `expertflow_cache_state`, eight authoritative current demands, ordered candidate IDs, candidate count, and `force_evict`.
- Produces: `expertflow_cache_projected_candidate` containing selected rank/expert, projected plan, and projected state.

- [ ] Add assertion-active tests proving a candidate admitted by the current projected plan is skipped, a later still-absent candidate is selected, live state is byte-identical, LRU updates affect selection, invalid inputs fail closed, and no-candidate is explicit.
- [ ] Run `test-expertflow-cache` and verify RED because the selector does not exist.
- [ ] Implement the helper using the existing planner, commit, and committed-mapping validator on a copied state.
- [ ] Run the focused test and verify GREEN.

### Task 2: Replace T2 admission filter

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test_t2_sidecar_source_contract.py`

**Interfaces:**
- Consumes: the temporal pending prediction and authoritative source-token IDs passed to `ggml_backend_sched_expertflow_temporal_observe_router`.
- Produces: one projected-absent candidate passed into the unchanged sidecar enqueue path.

- [ ] Add a failing source contract requiring the projected-state selector and forbidding direct current-state residency traversal in T2 enqueue.
- [ ] Run the source test and verify RED.
- [ ] Pass the current authoritative selected IDs into T2 enqueue and call the pure selector.
- [ ] Preserve the existing sidecar reservation, packed ranges, CUDA enqueue, reconciliation, and fallback.
- [ ] Add deferred policy fields to T2 records without allocation or formatting in the capture path.
- [ ] Run source and native tests and build.

### Task 3: Analysis of prevented misses

**Files:**
- Modify: `src/expertflow/predictor/temporal_sidecar_analysis.py`
- Modify: `tests/test_temporal_sidecar_analysis.py`
- Modify: `scripts/summarize_t2_temporal_sidecar.py`

**Interfaces:**
- Consumes: paired reactive/T2 cache logs and sidecar mappings.
- Produces: actual baseline misses covered, redundant baseline hits covered, and strict reconciliation totals.

- [ ] Add failing tests where a sidecar demand corresponds to a paired baseline miss and where it corresponds to a baseline hit.
- [ ] Run tests and verify RED.
- [ ] Implement event-aligned paired analysis without inferring misses from aggregate counters.
- [ ] Run tests and verify GREEN.

### Task 4: Controlled and focused runtime validation

**Files:**
- Reuse: `scripts/run_t2_temporal_sidecar_suite.py`
- Create: `docs/evidence/live-cache/t2-projected-state-policy-result.md`
- Create: `docs/evidence/live-cache/t2-projected-state-policy-summary.json`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Produces: exactness, determinism, transfer, prevented-miss, performance, memory, cleanup, and final runtime-freeze evidence.

- [ ] Build the isolated runtime and probe and record binary hashes.
- [ ] Run a focused smoke and stop on parity, ownership, mapping, or cleanup failure.
- [ ] Run one warmup plus three measured reactive/projected-policy pairs for general, code, and translation.
- [ ] Validate exact tokens/router projections and repeated determinism.
- [ ] Compute prevented misses, ready/late/waste, blocking, timing classes, TPS, latency, bytes, VRAM, and cleanup.
- [ ] Run the complete ExpertFlow suite, native tests, judge replay, build, and `git diff --check`.
- [ ] Commit llama.cpp implementation and ExpertFlow evidence separately with `Assisted-by: Codex`.
- [ ] Record runtime frozen and begin release integration only; do not start another runtime experiment.
