# Canonical One-Layer Blocking Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove exact true-router-directed blocking replacement for eight coordinated packed Gemma Q4 expert slots at layer 24 within the canonical Observer v1 runtime.

**Architecture:** Use the existing tensor-buffer override API to keep only layer-24 MoE expert tensors on host when the cache flag is enabled. The approved eight-slot arena makes every slot replaceable so each authoritative top-8 set can be loaded and remapped without a kernel or graph rewrite.

**Tech Stack:** C++17, CUDA 12.8, MSVC 19.39, CMake/Ninja, pinned llama.cpp `a7312ae9`, MinGW canonical ExpertFlow probe, Python 3.12/pytest.

## Global Constraints

- Protected Observatory and clean pinned checkout remain untouched.
- Worktrees: `C:\models\expertflow\worktrees\one-layer-blocking-cache` and `C:\models\expertflow\worktrees\llama-one-layer-blocking-cache`.
- Default `live_cache_enabled=false`; cache-off follows the existing canonical observer path.
- Layer 24 only; packed object 3,345,412 bytes, aligned contract 3,346,048 bytes.
- Eight coordinated replaceable physical slots are the approved minimum for unchanged top-8 `MUL_MAT_ID`.
- Blocking transfers only. No repacking, prediction, async stream, MTP, ML, multi-layer cache, allocator rewrite, or speed claim.
- Apply RED-GREEN-REFACTOR and preserve every command/result in the append-only ledger.

---

### Task 1: Freeze baseline and mapping evidence

**Files:**
- Modify: `PROJECT_LOG.md`
- Create: `docs/evidence/live-cache/canonical-one-layer-map.md`
- Create: `configs/canonical-one-layer-cache.json`

- [x] Record canonical hashes, commits, environment, prompt suite, layer placement, tensor names/types/shapes/strides, consumers, and the eight-slot minimum.
- [x] Run 89 ExpertFlow tests, verify canonical binary/model hashes, judge replay, and protected/pinned cleanliness.
- [x] Preserve the design/mapping evidence for the verified C4 milestone commit.

### Task 2: Create the isolated llama.cpp prototype and RED planner tests

**Files:**
- Create in llama worktree: `ggml/src/ggml-expertflow-cache.h`
- Create in llama worktree: `ggml/src/ggml-expertflow-cache.cpp`
- Create in llama worktree: `tests/test-expertflow-cache.cpp`
- Modify in llama worktree: `ggml/src/CMakeLists.txt`
- Modify in llama worktree: `tests/CMakeLists.txt`

- [x] Add tests for disabled configuration, strict layer-24 contract, eight unique selected IDs, replacement planning, hit reuse, forced eviction, invalid selection failure, byte bounds, and stable mappings.
- [x] Build and run the focused test to observe the expected missing-symbol RED failure; correct the Release `NDEBUG` false-green and rerun RED.
- [x] Implement the smallest pure planner/config parser needed to pass; run focused tests GREEN with assertions active.

**Authorization gate:** Passed on 2026-07-16. The user approved eight coordinated replaceable slots as the minimum exact architecture.

### Task 3: Add cache-enabled tensor placement and C0/C1

**Files:**
- Modify: `native/router_probe/main.cpp`
- Modify in llama worktree: `ggml/src/ggml-backend.cpp`

- [x] Add an ExpertFlow contract test that cache variables are absent by default and invalid combinations fail before model load; observe RED.
- [x] When explicitly enabled, pass exact layer-24 expert tensor overrides to CPU. When disabled, leave `llama_model_params` on the existing path.
- [x] Add scheduler-local fixed state with no arena use in passthrough mode.
- [x] Build the canonical probe against the modified runtime and pass C0/C1 canonical parity.

### Task 4: Allocate the eight-slot arena and prove C2

**Files:**
- Modify in llama worktree: `ggml/src/ggml-backend.cpp`
- Modify in llama worktree: `ggml/src/ggml-expertflow-cache.cpp`

- [x] Add RED tests for exact three-component destination layouts, aligned allocation totals, and out-of-range offsets.
- [x] Create the three reduced rank-3/scale destination layouts in one persistent scheduler-owned arena; never allocate per token.
- [x] Load an exact selected set directly from host Q4/F32 slices, synchronize, remap the eight IDs, and execute both matrix operations and scale lookup.
- [x] Require exact C2 tokens and logical router selections against C0.

### Task 5: Controlled replacement and C3

**Files:**
- Modify in llama worktree: `ggml/src/ggml-backend.cpp`
- Modify: `scripts/run_canonical_smoke.py`

- [x] Add tests for deterministic retention/replacement generations, zero-byte hits, and forced misses.
- [x] Implement blocking replacement only after the preceding graph use is synchronized.
- [x] Run normal multi-expert replacement and 41 consecutive forced all-eight replacements; reconcile contents, generations, and exact parity.

### Task 6: True-router blocking C4 and evidence

**Files:**
- Modify in llama worktree: `ggml/src/ggml-backend.cpp`
- Create: `docs/evidence/live-cache/one-layer-c4-result.md`
- Modify: `configs/canonical-one-layer-cache.json`
- Modify: `PROJECT_LOG.md`

- [x] Let authoritative selected IDs drive hit/miss planning and direct component copies for every member of the current top-8 set.
- [x] Emit bounded JSONL events with step/layer/logical expert/slot/hit/bytes/timing/reason/residents; fail on overflow.
- [x] Run the seven-task suite three times (21 fresh processes), forced replacements, memory/process cleanup, 89 ExpertFlow tests, focused native tests, feature-off restoration, and judge replay.
- [x] Hash the binary and evidence and write the exact C4 result. Commit only after final diff/test verification.
