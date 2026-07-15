# Non-Perturbing Router Observer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quarantine callback-derived real-model evidence and determine whether merely registering the llama.cpp evaluation callback changes clean-runtime output.

**Architecture:** Add a checked-in evidence-status manifest that withdraws callback-derived claims without deleting artifacts. Extend the separate router probe with one diagnostic `empty` callback mode, then run only T0/T1 first. If T1 fails, stop the in-callback matrix and document scheduler interception as the earliest perturbing boundary.

**Tech Stack:** Python 3.12/pytest, C++17, MinGW UCRT64, CMake/Ninja, PowerShell, JSON/JSONL, pinned llama.cpp `a7312ae94f801fc9c6786dc56e38df57b964f697`, clean MSVC/CUDA 12.8 runtime.

## Global Constraints

- Preserve commit `c41b9394c0443a66b3b486936d128c034ea3a4d7` and every existing artifact.
- Keep the protected Observatory and pinned llama.cpp checkout clean.
- Stop all corpus collection; label current callback traces `trace_v1_perturbing`.
- Do not use quarantined traces for final locality, policy, deadline, or Gate 4 claims.
- Keep `live_cache_enabled=false`; do not implement cache, MTP, async prefetch, or API work.
- Implement variants sequentially and stop after the first failing variant.
- Log commands, failures, timings, hashes, and decisions under `C:\models\expertflow\runs\trace-observer-repair`.

---

### Task 1: Quarantine callback-derived evidence

**Files:**
- Create: `configs/trace-evidence-status.json`
- Create: `tests/test_trace_evidence_status.py`
- Modify: `README.md`
- Modify: `docs/evidence/q4-deadline-oracle.md`
- Modify: `docs/evidence/q4-heldout-routing.md`
- Modify: `docs/evidence/q4-live-cache-go-no-go.md`
- Modify: `docs/evidence/q4-physical-feasibility-routing.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: real-model artifact roots and derived evidence named in the approved design.
- Produces: schema `1.0.0` manifest with `trace_generation`, `real_model_roots`, `derived_claims`, and `offline_fixture` fields.

- [ ] **Step 1: Write the failing quarantine contract test**

Create `tests/test_trace_evidence_status.py` that loads `configs/trace-evidence-status.json`, requires `trace_v1_perturbing`, requires all listed real-model roots and derived claims to have `excluded_from_final_claims=true`, requires `93.28%` to be withdrawn, and limits the replay fixture to `synthetic_or_offline_validation_only`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/test_trace_evidence_status.py -q`

Expected: FAIL because `configs/trace-evidence-status.json` does not exist.

- [ ] **Step 3: Add the minimal manifest and withdrawal notices**

List the Q4 baseline/GPU, stratified CUDA/Vulkan, held-out, physical-feasibility, and Gate 3 roots. Mark derived locality, static, LRU, session, oracle, and deadline results excluded. Add a prominent quarantine notice to each affected recommendation document and README without deleting historical tables.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_trace_evidence_status.py -q`

Expected: `1 passed`.

Run: `uv run pytest -q`

Expected: 88 tests pass.

- [ ] **Step 5: Commit the quarantine**

Run `git diff --check`, stage only Task 1 files, and commit `docs: quarantine perturbing router traces`.

### Task 2: Add the T1 empty-callback diagnostic

**Files:**
- Modify: `native/router_probe/main.cpp`
- Create externally: `C:\models\expertflow\runs\trace-observer-repair\test_trace_modes.ps1`
- Create externally: `C:\models\expertflow\runs\trace-observer-repair\runtime-clean`
- Create externally: `C:\models\expertflow\builds\trace-observer-probe`

**Interfaces:**
- Consumes: existing probe arguments and copied clean runtime DLLs.
- Produces: `--trace-mode full|empty|disabled`; existing `--trace` commands retain full behavior.

- [ ] **Step 1: Write and run the failing CLI contract**

The external PowerShell contract invokes the preserved c41 probe with `--help`, checks that `--trace-mode` is absent, then invokes `--trace-mode empty` with an intentionally missing model and requires argument rejection before model loading.

Expected RED: contract exits 1 because the new option is absent.

- [ ] **Step 2: Implement the minimum parser and empty callback**

Add `enum class trace_mode { full, empty, disabled };`. Parse exactly `full`, `empty`, or `disabled`. `--no-trace` selects disabled. Full mode requires `--trace`; empty/disabled reject a trace path and capture flags. Add:

```cpp
bool empty_trace_callback(ggml_tensor *, bool, void *) {
    return false;
}
```

Register `router_trace_callback` only for full mode, `empty_trace_callback` only for empty mode, and no callback for disabled mode. Run trace state transitions and zero-event validation only in full mode.

- [ ] **Step 3: Build in a new runtime copy**

Copy the 90 clean-runtime files into `runtime-clean`, configure the probe with `C:\msys64\ucrt64\bin\g++.exe`, Ninja, the pinned headers, the existing read-only import libraries, and the new runtime path. Build without overwriting Gate 3 audit runtimes.

- [ ] **Step 4: Verify GREEN and backward compatibility**

Run the CLI contract against the new probe. Require help exposure, invalid-value rejection, empty mode reaching missing-model load, and unchanged full/disabled parsing. Run one no-model contract for each mode and hash the probe/runtime files.

- [ ] **Step 5: Commit the diagnostic mode**

Run `git diff --check`, stage only `native/router_probe/main.cpp`, and commit `test: add empty router callback mode`.

### Task 3: Execute the T0/T1 isolation matrix

**Files:**
- Create externally: `C:\models\expertflow\runs\trace-observer-repair\run_t0_t1_matrix.py`
- Create externally: `C:\models\expertflow\runs\trace-observer-repair\matrix-summary.json`

**Interfaces:**
- Consumes: exact clean runtime/probe, Q4 model, frozen general/code/translation prompts.
- Produces: 18 measured runs: three disabled and three empty-callback runs per domain.

- [ ] **Step 1: Implement a bounded measured runner**

Reuse `run_native_measured.py` without modifying it. Emit per-run command, hashes, duration, token artifact, GPU before/peak/settled, process peak memory, return code, and mode. Do not request router tensors or trace files.

- [ ] **Step 2: Run all T0/T1 repetitions**

Use greedy sampling, 16 generated tokens, ten GPU layers, 12 threads, and identical prompt bytes. Abort on native failure but retain completed manifests.

- [ ] **Step 3: Compare exact outputs and timing**

For each domain, require three T0 token files to match, three T1 token files to match, and compare T0 versus T1. Record first generated-token mismatch, duration median/range, peak/settled memory, and persistent process count.

- [ ] **Step 4: Apply the early-stop rule**

If any domain fails T0/T1 token parity, classify T1 as the first failing variant and do not implement T2-T7 in this callback. If all pass, return to the approved design and create a separate T2 plan.

### Task 4: Root-cause report and checkpoint

**Files:**
- Create: `docs/evidence/live-cache/trace-observer-isolation.md`
- Modify: `configs/trace-evidence-status.json`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: source audit, matrix summary, runtime/probe hashes, and quarantine manifest.
- Produces: explicit first-failing variant, decision, next instrumentation boundary, and unchanged Gate 4 state.

- [ ] **Step 1: Document scheduler-level causality**

If T1 fails, show that the empty callback performs no readback/allocation/I/O/state mutation, while non-null `cb_eval` selects the synchronized graph-view scheduler branch. Separate measured token/timing results from source-derived causality.

- [ ] **Step 2: Verify the checkpoint**

Read and use `superpowers:verification-before-completion`. Run the full ExpertFlow suite, judge replay, JSON/JSONL parsing, artifact-hash reconciliation, `git diff --check`, worktree/protected/source cleanliness, no cache environment variables, and no persistent model process.

- [ ] **Step 3: Commit the isolation result**

Commit `docs: record first perturbing trace variant`. Do not issue a Gate 4 recommendation or collect replacement traces unless the complete new tracing gate passes.
