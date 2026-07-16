# P2 layer-24 asynchronous prefetch implementation plan

> Execute sequentially. Do not begin a later stage until the previous stage's
> evidence and stop conditions have been reviewed.

**Goal:** Add a disabled-by-default, exact, bounded asynchronous predictive
prefetch path for the existing layer-24 32-slot packed-Q4 cache.

**Architecture:** Keep policy and slot ownership in the common scheduler. Add a
small opaque CUDA transfer service behind backend registry function pointers.
Use a fixed slot state machine, fixed descriptors, fixed pinned staging, one
dedicated stream, and completion events. The true router and existing reactive
blocking cache remain authoritative.

**Repositories:** ExpertFlow evidence/scripts in
`C:\models\expertflow\worktrees\p2-layer24-async-prefetch`; runtime changes in
`C:\models\expertflow\worktrees\llama-p2-layer24-async-prefetch`.

---

## Task 1: Freeze and reproduce the combined baseline

**Files:**

- Modify: `PROJECT_LOG.md`
- Create: `docs/evidence/predictor/p2-baseline.md`
- Create runtime artifacts under
  `C:\models\expertflow\runs\p2-layer24-async-prefetch\baseline`

1. Record both repository heads, merge bases proving C5 ancestry, status,
   submodules, compiler/CUDA/driver versions, environment, model hash, build
   configuration, and executable hashes.
2. Build the unmodified P1+C5 llama.cpp worktree and matching router probe.
3. Run ExpertFlow tests and targeted llama.cpp predictor/cache tests.
4. Reproduce prediction-disabled/cache-disabled behavior.
5. Reproduce C5 layer-24 32-slot reactive exactness on fixed general, code, and
   translation prompts with at least three repetitions.
6. Reproduce P1 offline/live-shadow output equivalence with cache disabled.
7. Preserve every command and raw result. Stop on parity, routing, memory, or
   cleanup failure.

## Task 2: Test-drive the slot lifecycle and admission planner

**Files:**

- Modify: `ggml/src/ggml-expertflow-cache.h`
- Modify: `ggml/src/ggml-expertflow-cache.cpp`
- Modify: `tests/test-expertflow-cache.cpp`

1. Add failing tests for legal and illegal slot-state transitions.
2. Add failing tests for generation, ownership, bounds, and incomplete-transfer
   rejection.
3. Add failing tests proving admission planning does not mutate cache state.
4. Add failing tests for empty-slot-only admission, unsafe-victim rejection,
   deterministic victim selection, and one-transfer budget enforcement.
5. Implement the minimum pure-CPU state machine and planner to pass.
6. Run the focused cache test after each red/green step.

## Task 3: Add a CUDA opaque transfer service

**Files:**

- Create: `ggml/src/ggml-cuda/expertflow-prefetch.cuh`
- Create: `ggml/src/ggml-cuda/expertflow-prefetch.cu`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify source-contract tests in ExpertFlow or llama.cpp as appropriate

1. Add source-contract tests for the required registry functions, dedicated
   non-blocking stream, pinned staging, fixed descriptors, events, and cleanup.
2. Define an opaque service API for create, destroy, stage/queue, query, wait,
   and metric retrieval.
3. Preallocate all descriptors, events, and pinned storage during create.
4. Queue three component copies and record start/end/completion events on the
   dedicated stream.
5. Expose the functions through `ggml_backend_cuda_reg_get_proc_address`.
6. Build CUDA and run the contract and unit tests.

## Task 4: P2.0 one-known-expert asynchronous proof

**Files:**

- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `native/router_probe/live_cache_config.h`
- Modify: `native/router_probe/live_cache_config.cpp`
- Modify: `native/router_probe/main.cpp`
- Modify matching configuration tests
- Create: `scripts/run_p2_async_transfer_proof.py`

1. Add a disabled-by-default P2 feasibility configuration.
2. Resolve the CUDA transfer service only for an eligible layer-24 blocking
   cache.
3. Add fixed scheduler-owned transfer metadata without changing execution.
4. Queue one configured known expert into a safe unused slot.
5. Return without compute-stream synchronization, run normal intervening work,
   and reconcile later.
6. Verify exact source/destination ranges and copied bytes before allowing the
   proof to pass.
7. Measure staging, enqueue, CPU return, CUDA-event H2D, queue-to-ready host
   wall, and overlap window separately.
8. Run exactness, repetition, memory, and cleanup checks. Stop if genuine
   asynchronous behavior or safe lifetime cannot be proven.

## Task 5: P2.1 admission planning without mutation

**Files:**

- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-expertflow-predictor.*` only if a narrow immutable
  candidate handoff is required
- Modify predictor/cache tests

1. Permit the accepted P1 predictor to coexist with layer-24 cache only under
   explicit P2 modes.
2. At the canonical layer-23 event, compute the frozen candidates.
3. Build and record an admission plan without reserving, copying, or remapping.
4. Verify deterministic plans and exact cache/reactive outputs across the
   focused suite.

## Task 6: P2.2 at most one predicted transfer

**Files:**

- Modify: `ggml/src/ggml-backend.cpp`
- Modify: `ggml/src/ggml-expertflow-cache.*`
- Modify focused unit/source-contract tests
- Create: `scripts/run_p2_predictive_validation.py`

1. Reserve one safe slot and generation from the immutable admission plan.
2. Stage and queue exactly one predicted expert.
3. Reconcile completion at authoritative layer 24.
4. Accept only matching ready generations; classify late and unused transfers.
5. Fall back to the unchanged reactive blocking loader for all unmet demand.
6. Preserve true IDs, order, weights, and final verified mapping.
7. Test ready, late, unused, stale-generation, forced-miss, repeated-swap,
   overflow, and teardown cases.
8. Run three repetitions per focused prompt and mode; stop immediately on any
   exactness or stability failure.

## Task 7: Decide and, if justified, run P2.3

**Files:**

- Modify only the bounded transfer-budget configuration and tests
- Create: `docs/evidence/predictor/p2-concurrency-decision.md`

1. Summarize P2.0/P2.2 transfer duration, overlap window, ready/late rate,
   wasted bytes, staging capacity, VRAM, and pinned-host use.
2. Freeze either no expansion or one small concurrency budget.
3. If expanded, repeat all exactness, memory, cleanup, and focused performance
   checks. Do not search multiple widths or budgets for a favorable result.

## Task 8: Final validation and evidence

**Files:**

- Modify: `README.md` only if user-facing reproduction changes
- Modify: `PROJECT_LOG.md`
- Create: `docs/evidence/predictor/p2-result.md`
- Create/update machine-readable evidence under `evidence/`

1. Compare cache-disabled, C5 reactive, and P2 predictive modes using identical
   settings and repetitions.
2. Report tokens, routing, prompt TPS, decode TPS, end-to-end time, TTFT and
   p50/p95 token latency where captured, peak VRAM, blocking time, transfers,
   bytes, hit/ready/late/waste rates, CUDA-event latency, and host timings.
3. Run the complete ExpertFlow suite, judge replay, deterministic model checks,
   focused prompt suite, memory/process cleanup, and worktree status checks.
4. Hash binaries, configs, raw artifacts, and reports.
5. State pass, bounded partial result, or stop without a speedup claim unless
   measured end-to-end decode TPS improves against identical reactive C5.
6. If P2 passes, commit llama.cpp and ExpertFlow milestones separately with
   `Assisted-by: Codex` trailers. Do not merge or push.
