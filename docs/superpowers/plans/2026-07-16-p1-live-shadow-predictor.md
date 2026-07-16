# P1 Live Shadow Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the frozen B2 layer-23-to-layer-24 predictor inside the canonical runtime in telemetry-only shadow mode.

**Architecture:** ExpertFlow deterministically exports one versioned binary table artifact and offline parity fixtures. llama.cpp loads the artifact once into fixed-capacity arrays, receives an explicit one-call phase from the probe, predicts after layer 23, and reconciles against layer 24 without moving weights or changing cache state.

**Tech Stack:** Python 3.12, C++17, MSVC v143, CUDA 12.8 build, pytest, CMake/Ninja, SHA-256, JSONL.

## Global Constraints

- Frozen P0: commit `6bc8eb68`, source-normalized phase-separated B2, width 12, observed-support admission.
- Do not rerun the sealed test, retune, train, change width, or change admission.
- Single transition only: layer 23 to layer 24.
- Shadow mode only; no expert transfers, cache admissions, reservations, evictions, async streams, higher `-ngl`, 64 slots, MTP, or MLP work.
- Predictor and live cache remain disabled by default.
- No per-prediction allocation, file I/O, formatting, graph callback, tensor request, or added synchronization.
- llama.cpp commits require explicit user authorization after P1 passes.

---

### Task 1: Deterministic artifact exporter

**Files:**
- Create: `src/expertflow/predictor/runtime_artifact.py`
- Create: `scripts/export_runtime_predictor.py`
- Create: `tests/test_predictor_runtime_artifact.py`

**Interfaces:**
- Consumes: frozen `b2_source_normalized_separate.pkl`, selection lock, and expanded manifest.
- Produces: `expertflow-b2-23-24-v1.bin`, metadata JSON, and deterministic parity-fixture JSON.

- [ ] Write failing tests for deterministic bytes, exact header identifiers, payload checksum, table dimensions, float64 values, support masks, fallback vector, and fixture candidate order.
- [ ] Run `python -m pytest tests/test_predictor_runtime_artifact.py -q` and confirm failures because the exporter is absent.
- [ ] Implement fixed little-endian structs, validation of the frozen lock/artifact hashes, source-normalized table export for layer 24, and deterministic fixture generation.
- [ ] Run the focused tests green, export twice, and require identical SHA-256.

### Task 2: Native artifact loader and scorer

**Files:**
- Create: `ggml/src/ggml-expertflow-predictor.h`
- Create: `ggml/src/ggml-expertflow-predictor.cpp`
- Create: `tests/test-expertflow-predictor.cpp`
- Modify: `ggml/src/CMakeLists.txt`
- Modify: root or test CMake registration only as required by existing patterns.

**Interfaces:**
- Consumes: artifact path, explicit phase enum, eight source IDs.
- Produces: twelve IDs, twelve float64 scores, latency, artifact/config hashes.

- [ ] Add RED native tests for fixture agreement, both phases, ascending-ID ties, fallback, repeated determinism, and every fail-closed artifact condition.
- [ ] Build the test and confirm it fails because the loader/scorer is absent.
- [ ] Implement one-time load into fixed-size arrays and a stack-only prediction function.
- [ ] Build and run native tests green under assertion-active Release.

### Task 3: One-call phase contract

**Files:**
- Modify: `include/llama.h`
- Modify: the narrow llama context/scheduler bridge required by the existing decode path.
- Modify: `native/router_probe/main.cpp`
- Create or modify focused phase tests in both repositories.

**Interfaces:**
- Produces: explicit `unset/prefill/decode` phase plus monotonically increasing generation scoped to one `llama_decode` call.

- [ ] Add RED tests for prefill/decode, unset, invalid, contradictory set, stale generation, reset after success/failure, and repeated calls.
- [ ] Add the smallest API and scheduler state bridge; set immediately before and clear immediately after the probe's decode call.
- [ ] Run focused tests green and prove disabled mode does not require the setter.

### Task 4: Fixed-capacity shadow capture

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `tests/test_multilayer_cache_source.py`
- Modify: `native/router_probe/main.cpp` only for configuration and phase/run identity.

**Interfaces:**
- Consumes: layer-23 and layer-24 already-materialized host IDs.
- Produces: deferred P1 JSONL records without cache mutation.

- [ ] Add RED source/native tests requiring disabled-by-default behavior, one pending transition, fixed-capacity completed records, strict order/generation checks, and deferred serialization.
- [ ] Implement environment parsing for explicit artifact/output paths and the single transition.
- [ ] Predict at layer 23, reconcile at layer 24, compute recall contributions, and append fixed records.
- [ ] Add teardown serialization and explicit overflow/incomplete-transition failures.
- [ ] Run source/native tests green and rebuild the CUDA runtime.

### Task 5: Offline/live equivalence harness

**Files:**
- Create: `src/expertflow/predictor/live_shadow.py`
- Create: `scripts/run_live_shadow_validation.py`
- Create: `tests/test_live_shadow_validation.py`

**Interfaces:**
- Consumes: runtime JSONL, canonical routing trace, parity fixtures, tokens, and performance artifacts.
- Produces: exact equivalence, parity, latency, recall, overhead, memory, and cleanup summaries.

- [ ] Add RED tests for candidate/score equality, transition identity, phase counts, token/router parity, latency percentiles, and missing/duplicate events.
- [ ] Implement strict reconciliation and machine-readable summaries.
- [ ] Run focused tests green.

### Task 6: S0-S3 measured validation

**Files:**
- Create: `configs/p1-live-shadow.json`
- Create: `docs/evidence/live-cache/p1-live-shadow-result.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Produces: measured general/code/translation and seven-task evidence.

- [ ] Run S0 disabled and S1 load-only parity.
- [ ] Run S2 fixture/native verification.
- [ ] Run S3 one warmup plus three measured repetitions for focused prompts, then the seven-task smoke suite.
- [ ] Verify exact tokens, routers, offline/live candidates, phases, counts, determinism, graph shape, memory, cache immutability, zero predictor transfers, cleanup, and overhead.
- [ ] Run the complete ExpertFlow, replay, llama source/native, and reproduction suites.
- [ ] Stop on any P1 stop condition.

### Task 7: Passing milestone and P2 handoff

**Files:**
- Create: `docs/superpowers/specs/2026-07-16-p2-async-layer24-design.md` only after P1 passes.

- [ ] Hash the runtime binary, predictor artifact, configuration, and measured summaries.
- [ ] Request explicit authorization for the llama.cpp and ExpertFlow P1 commits.
- [ ] Commit only after authorization, keep branches unmerged and unpushed, and do not implement P2.
