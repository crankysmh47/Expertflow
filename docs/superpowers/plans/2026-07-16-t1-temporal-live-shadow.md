# T1 Temporal Live-Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the frozen T0 layer-24 next-token predictor in the live canonical runtime and measure the real shadow-only transfer lead-time window.

**Architecture:** Export a distinct hashed temporal binary artifact and golden vectors. Add a fixed native scorer/state machine and a disabled-by-default scheduler integration that observes authoritative layer-24 decode IDs, records host-wall timing, and never touches cache residency or CUDA transfer services.

**Tech Stack:** Python 3.12, C++17, MSVC v143, CUDA 12.8 build, pinned llama.cpp lineage `d8354b17`, pytest, CMake/Ninja, JSONL, SHA-256.

## Global Constraints

- Preserve T0 commit `afe17d1`, P2 llama.cpp commit `d8354b17`, and all protected branches.
- Layer 24 and decode only; relationship is token `t` to token `t+1`.
- Frozen weights are exactly `0.5 / 0.4 / 0.1`; candidate width is exactly 16.
- No cache mutation, weight movement, CUDA prefetch, graph segmentation, synchronization, or T2 work.
- Feature disabled by default; `live_cache_enabled=false`.
- No speedup or overlap claim.

---

### Task 1: Temporal runtime artifact and golden vectors

**Files:**
- Create: `src/expertflow/predictor/temporal_runtime_artifact.py`
- Create: `scripts/export_temporal_runtime_predictor.py`
- Create: `tests/test_temporal_runtime_artifact.py`
- Create: `tests/test_temporal_runtime_export.py`

**Interfaces:**
- Produces `build_temporal_runtime_artifact`, `parse_temporal_runtime_artifact`,
  and `predict_temporal_runtime_artifact`.
- Produces a `.bin`, metadata JSON, and golden-vector JSON under the T1 run
  root.

- [ ] Write RED tests for magic/version/identity/hash validation, exact scorer
  reproduction, non-finite rejection, decode-only behavior, sixteen candidates,
  session update order, and deterministic export.
- [ ] Run the focused tests and confirm missing-module failures.
- [ ] Implement the fixed artifact and exporter using the committed T0 lock and
  selected pickle; refuse any different policy, weights, width, manifest, or
  artifact hash.
- [ ] Export at least sixteen validation golden transitions spanning multiple
  conversations and verify parse/predict equality against offline rankings and
  float64 scores.
- [ ] Run focused and full Python tests.

### Task 2: Native temporal scorer and state

**Files:**
- Create: `ggml/src/ggml-expertflow-temporal.h`
- Create: `ggml/src/ggml-expertflow-temporal.cpp`
- Create: `tests/test-expertflow-temporal.cpp`
- Modify: `ggml/src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces fixed artifact load, scorer, reset, observe, and completed-record
  functions with no heap allocation in observation.

- [ ] Write RED native tests for artifact identity, golden vectors, first-token
  seed behavior, exact session counts, consecutive forward/decode identity,
  prefill rejection, duplicate/skip rejection, generation reset, overflow, and
  non-finite/invalid data.
- [ ] Build the test target and confirm compile/link failures from missing API.
- [ ] Implement the minimal loader, scorer, and fixed state machine.
- [ ] Run native tests against the exported artifact and golden vectors.

### Task 3: Disabled-by-default scheduler integration

**Files:**
- Modify: `ggml/include/ggml-backend.h`
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `include/llama.h`
- Modify: `src/llama-context.cpp`
- Modify: `native/router_probe/import/llama.def`
- Modify: `native/router_probe/main.cpp`
- Modify: `tests/test-expertflow-temporal.cpp`

**Interfaces:**
- Adds explicit temporal reset and layer-24 observation APIs.
- Environment: `EXPERTFLOW_TEMPORAL_SHADOW=1`,
  `EXPERTFLOW_TEMPORAL_ARTIFACT`, `EXPERTFLOW_TEMPORAL_SHADOW_LOG`, and
  `EXPERTFLOW_TEMPORAL_RUN_ID`.

- [ ] Write RED source/native tests for exact environment parsing, required
  paths, cache-disabled requirement, reset-before-observe, layer/phase
  rejection, fixed storage, deferred output, and no P2 CUDA service use.
- [ ] Implement scheduler-owned temporal state and teardown serialization.
- [ ] Call explicit reset once before canonical inference and observe only
  layer-24 decode IDs already materialized by the canonical callback.
- [ ] Record source/predictor/target timestamps, IDs, scores, session counts,
  ranking matches, generation, and summary without file I/O in observation.
- [ ] Build llama.cpp and router probe; run native tests.

### Task 4: Validation and analysis

**Files:**
- Create: `scripts/run_t1_temporal_shadow_suite.py`
- Create: `scripts/summarize_t1_temporal_shadow.py`
- Create: `tests/test_t1_temporal_shadow.py`
- Create: `tests/test_t1_temporal_summary.py`

**Interfaces:**
- Produces paired disabled/enabled manifests and a summary containing exactness,
  lead-time distributions, ranking metrics, one-transfer simulation, memory,
  overhead, and cleanup.

- [ ] Write RED parser/summary tests for missing/duplicate/skipped transitions,
  exact offline/live IDs/scores/counts, percentile math, deadline classes,
  hit@1/2/4, recall, MRR, reactive LRU admission, usefulness, waste, regret,
  parity, and repetition aggregation.
- [ ] Implement a runner using the existing fixed general/code/translation
  methodology, one warmup and three measured repetitions per mode.
- [ ] Implement strict validation and summary with reference costs labeled
  separately from live timestamps.
- [ ] Run the focused suite; stop on any parity, equivalence, state, memory, or
  overhead failure.

### Task 5: Evidence, verification, and gated commit

**Files:**
- Create: `docs/evidence/live-cache/t1-temporal-live-shadow.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Produces a written T1 pass/stop verdict and a separate one-transfer T2
  recommendation only when acceptance criteria pass.

- [ ] Report repetition-level exactness, prediction latency, lead time,
  deadline eligibility, ranking/usefulness, overhead, VRAM/host memory, reset,
  and cleanup.
- [ ] Run complete ExpertFlow tests, native temporal/predictor/cache tests,
  judge replay, `git diff --check`, artifact hashes, and process cleanup.
- [ ] If T1 passes, commit llama.cpp runtime changes and ExpertFlow
  evidence/tools separately with `Assisted-by: Codex`; keep both unmerged and
  unpushed.
- [ ] Do not begin T2 automatically.

