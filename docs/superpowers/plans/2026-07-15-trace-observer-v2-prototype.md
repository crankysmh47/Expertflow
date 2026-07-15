# Trace Observer v2 One-Layer Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that one configured Gemma 4 MoE layer's already-materialized selected expert IDs can be recorded without evaluation callbacks, graph segmentation, new synchronization, token changes, or persistent memory growth.

**Architecture:** Add a private, disabled-by-default observer state to the pinned llama.cpp scheduler in `ggml-backend.cpp`. Configuration and fixed storage are created before execution; the hot capture path only validates fixed tensor metadata and copies scalar values already present in the existing host `ids` vector. Records are emitted only when the scheduler is destroyed.

**Tech Stack:** C++17, pinned llama.cpp `a7312ae94f801fc9c6786dc56e38df57b964f697`, MSVC 19.39, CUDA 12.8.93, CMake/Ninja, Python 3.12, pytest.

## Global Constraints

- Modify only `ggml/src/ggml-backend.cpp` in the llama.cpp prototype worktree.
- Keep the clean pinned checkout and protected Observatory untouched.
- Do not register or use `callback_eval`; do not request a tensor.
- Do not add graph views, synchronization, allocation, file I/O, formatting, or locking to the capture path.
- Preallocate a bounded record array during scheduler construction; stop computation explicitly on overflow.
- Restrict the prototype to one configured layer, top-k 8, I32 IDs, and one-token microbatches.
- Keep Gate 4 closed, `live_cache_enabled=false`, corpus collection stopped, prior traces quarantined, and the `93.28%` claim withdrawn.
- Stop on any token difference, routing difference, unexpected path absence, buffer failure, unexplained material overhead, or need for broader source changes.

---

### Task 1: Establish the RED observer contract

**Files:**
- Inspect: `C:/models/expertflow/worktrees/llama-trace-observer-v2/ggml/src/ggml-backend.cpp`
- Generate externally: `C:/models/expertflow/runs/trace-observer-v2-prototype/contracts/red.json`

**Interfaces:**
- Consumes: exact unmodified pinned source.
- Produces: a failing source contract requiring the observer mode enum, bounded state, environment prefix, overflow status, and capture call at the existing ID materialization boundary.

- [ ] **Step 1: Run a source-contract assertion before implementation**

Require these absent symbols: `expertflow_observer_v2_mode`, `expertflow_observer_v2_state`, `EXPERTFLOW_OBSERVER_V2_MODE`, `expertflow_observer_v2_capture`, and `GGML_STATUS_FAILED` at the capture call.

- [ ] **Step 2: Verify RED**

Run the assertion against the unmodified worktree. Expected: failure because the feature is absent, while the worktree remains clean.

### Task 2: Implement the minimal fixed-capacity observer

**Files:**
- Modify: `C:/models/expertflow/worktrees/llama-trace-observer-v2/ggml/src/ggml-backend.cpp:774-828`
- Modify: `C:/models/expertflow/worktrees/llama-trace-observer-v2/ggml/src/ggml-backend.cpp:1604-1621`
- Modify: `C:/models/expertflow/worktrees/llama-trace-observer-v2/ggml/src/ggml-backend.cpp:1727-1818`

**Interfaces:**
- Consumes: `ids_tensor`, the already-populated `std::vector<int32_t> ids`, and `prev_ids_tensor`.
- Produces: modes `noop`, `metadata`, and `ids`; one fixed record per target-layer decode step; a deferred JSONL artifact at scheduler destruction.

- [ ] **Step 1: Add private POD state and strict configuration**

Use a heap-allocated state pointer owned by the existing calloc-created scheduler. Parse mode, layer, output path, and capacity only in scheduler creation. Cap capacity at a small fixed maximum and allocate the record array once.

- [ ] **Step 2: Add the capture function**

Match `ffn_moe_topk-{layer}`, require I32/top-k 8/one row, write the current target-layer record ordinal as the decode-step index, and copy eight values only in `ids` mode. `noop` writes nothing. On capacity exhaustion, set overflow and return false without overwriting memory.

- [ ] **Step 3: Integrate after the existing ID materialization**

Call capture only inside `ids_tensor != prev_ids_tensor`, after lines 1604-1607 have populated `ids`. Return `GGML_STATUS_FAILED` on overflow/contract failure. Add no synchronization.

- [ ] **Step 4: Add deferred teardown emission**

At scheduler destruction, write a header/summary and records to the configured external path, report configuration errors explicitly, free storage, and leave disabled mode with a null state and zero side effects.

- [ ] **Step 5: Verify GREEN source contract**

Rerun the Task 1 assertion. Expected: pass, exactly one llama.cpp source file modified, and `git diff --check` clean.

### Task 3: Build and execute V0-V3

**Files:**
- Generate externally: `C:/models/expertflow/builds/llama-trace-observer-v2`
- Generate externally: `C:/models/expertflow/runs/trace-observer-v2-prototype/v0-v3/`

**Interfaces:**
- Consumes: the one-file prototype and existing clean CUDA build configuration.
- Produces: binary hashes plus V0 disabled, V1 noop, V2 metadata, and V3 IDs results.

- [ ] **Step 1: Configure and build with the supported toolchain**

Reuse the Gate 3 CMake cache values: Release, CUDA on, native off, CUDA architecture `120a-real`, tests/examples on, curl off. Record configure/build commands, versions, hashes, and duration.

- [ ] **Step 2: V0 compiled but disabled**

Run the frozen deterministic general/code/translation prompts three times with all observer variables unset. Require exact tokens versus the current clean baseline, no observer artifact, and stable memory/process cleanup.

- [ ] **Step 3: V1 noop**

Set mode `noop`, one layer, capacity, and output. Require exact V0 tokens, zero records, no callback, and no overflow.

- [ ] **Step 4: V2 metadata**

Set mode `metadata`. Require exact V0 tokens and one ordered layer/step record per forward.

- [ ] **Step 5: V3 selected IDs**

Set mode `ids`. Require exact V0 tokens, eight in-range IDs per record, identical ordered records across three repetitions, correct counts, no overflow/canary failure, and no persistent allocation/process growth.

- [ ] **Step 6: Compare overhead**

Report per-prompt/per-mode durations and median V3-versus-V0 overhead. Treat it as measured observer overhead, not model speedup.

### Task 4: Regression, evidence, and milestone decision

**Files:**
- Modify: `PROJECT_LOG.md`
- Modify: `configs/trace-evidence-status.json`
- Create: `docs/evidence/live-cache/trace-observer-v2-prototype.md`
- Generate externally: `C:/models/expertflow/runs/trace-observer-v2-prototype/verification.json`

**Interfaces:**
- Consumes: all V0-V3 artifacts and hashes.
- Produces: a pass/fail report and, only on full V3 success, a separate llama.cpp prototype commit and ExpertFlow evidence commit.

- [ ] **Step 1: Run regression gates**

Run all ExpertFlow tests, judge replay, existing deterministic model checks, JSON/JSONL parsing, `git diff --check`, protected/source cleanliness, live-cache environment checks, and persistent process checks.

- [ ] **Step 2: Apply the stop rule**

If any V-stage fails, preserve evidence and stop without commit expansion. If V3 passes, commit only `ggml/src/ggml-backend.cpp` in the llama prototype branch, then commit the evidence files in `codex/trace-observer-v2`.

- [ ] **Step 3: Issue the next recommendation without implementing it**

Recommend in order: all MoE layers for one token, one full prompt, then the frozen multi-domain suite. Do not begin any expansion, cache work, or replacement corpus collection.
