# Layer-24 C5 Reactive Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify an exact conversation-local 32-slot reactive LRU cache for layer 24 using the unchanged packed CUDA `MUL_MAT_ID` path.

**Architecture:** Generalize C4's pure planner and persistent direct CUDA operand to 32 resident slices while keeping each top-8 demand exact. Deterministic protected LRU selects slots; blocking transfers complete and validate before physical IDs replace logical IDs for execution.

**Tech Stack:** C++17, MSVC 19.39, CUDA 12.8, CMake/Ninja, pinned llama.cpp lineage `e41f54b0`, Python 3.12/pytest.

## Global Constraints

- Work only in `C:\models\expertflow\worktrees\c5-reactive-cache`, `C:\models\expertflow\worktrees\llama-c5-reactive-cache`, and `C:\models\expertflow\runs\c5-reactive-cache`.
- Exactly layer 24 and 32 direct packed CUDA slices; no eight-slice staging operand.
- Disabled by default; blocking true-router demand only.
- Preserve logical order/weights and exact parity; no prediction, async, MTP, ML, multilayer work, repacking, kernel/graph/general-allocator redesign, or speed claim.
- Apply RED-GREEN-REFACTOR and append commands, results, failures, measurements, artifacts, and decisions to the ledger.

---

### Task 1: Pure 32-slot protected-LRU planner

**Files:**
- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`

**Interfaces:**
- Consume: authoritative `std::array<int32_t, 8>` and monotonically increasing invocation sequence.
- Produce: 32-slot state plus an eight-demand plan containing physical IDs, hits, deterministic loads/evictions, generation changes, and committed recency.

- [x] Add assertion-active tests for lowest-free allocation, hit retention, demanded-slot protection, selected-order miss processing, LRU eviction, ascending-slot tie-break, reset, invalid selections, forced misses, copy bounds 0–31, and the 32-slot aligned projection.
- [x] Run the focused test and confirm RED from the missing 32-slot behavior.
- [x] Implement the minimum planner/layout changes; commit state only after a complete valid plan.
- [x] Rebuild and run the focused test GREEN, then run the existing C4 planner regressions.

### Task 2: Direct persistent 32-slice runtime operand

**Files:**
- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `native/router_probe/live_cache_config.cpp` only if a narrow C5 mode declaration is required.
- Modify: `native/router_probe/test_live_cache_config.cpp` only with a preceding RED test.

**Interfaces:**
- Consume: Task 1's physical IDs and copy plan.
- Produce: one persistent three-component 32-slice CUDA operand used directly by the unchanged layer-24 `MUL_MAT_ID` and scale path.

- [x] Add RED tests for strict C5 configuration and exact component destinations/bounds; verify the failure.
- [x] Generalize only the blocking layer-24 duplicated tensor layouts and arena offsets from 8 to 32; allocate once and preserve passthrough/disabled behavior.
- [x] Blocking-copy missing packed components, synchronize, validate completion/generations/full mapping, then publish physical IDs; abort explicitly on any failed invariant.
- [x] Extend bounded teardown-time JSONL records with LRU/recency and final 32-slot residents without adding hot-path I/O or per-token allocation.
- [x] Build the CUDA runtime and run C5-0/C5-1 exact parity before direct-cache execution.

### Task 3: C5-2 through C5-4 validation and milestone

**Files:**
- Create: `docs/evidence/live-cache/one-layer-c5-reactive-result.md`
- Modify: `configs/canonical-one-layer-cache.json`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consume: the exact C4 prompt/runtime evidence and C5 event logs.
- Produce: machine-readable comparisons and one written measured C5 verdict.

- [x] Run a known set twice and reconcile eight initial misses followed by hits, exact bytes, mappings, tokens, router order/weights, and stable allocation.
- [x] Run deterministic replacement plus forced misses and verify LRU/tie-break decisions, generations, no stale execution, and exact parity.
- [x] Run the focused deterministic prompt suite at least three times cache-off/cache-on; compare tokens, complete routing, order/weights where available, event counts/ordering, mappings, memory, determinism, cleanup, and feature-off restoration.
- [x] Run focused native/llama tests, all 89 ExpertFlow tests, judge replay, JSON/JSONL validation, `git diff --check`, and artifact hashing.
- [x] Write the exact measured allocation/transfer/memory result, keep timing diagnostic, and commit only if every gate passes.
