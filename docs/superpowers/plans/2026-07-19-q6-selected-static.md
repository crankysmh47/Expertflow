# Q6 Selected Static Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether static CUDA expert islands for Q6 layers `[0,1,15,20]` produce a stable, quality-safe, statistically meaningful decode improvement.

**Architecture:** Start from the frozen Q1b ExpertFlow and llama.cpp commits in separate worktrees. Add only a disabled stock split-timing probe for the bottleneck gate, then use the existing four-layer persistent-shadow implementation unchanged for static proof, matched cold-process performance, and frozen Q1b quality evaluation.

**Tech Stack:** Python 3.11, pytest, C++17, llama.cpp `a7312ae94`, MSVC v143, CUDA 12.8, CMake/Ninja, PowerShell, Gemma 4 26B A4B Q6_K.

## Global Constraints

- Preserve official Q1 failure and Q1b pass.
- Do not download a model or implement reactive caching, eviction, reduced capacity, prediction, new kernels, scheduler redesign, whole-layer placement, Q8, or product CLI work.
- Heavy processes run sequentially, hidden, and with file-only logs.
- Stop at the first declared gate failure.
- Keep worktrees unmerged and unpushed; do not commit the GGUF.

---

### Task 1: Stock bottleneck probe

**Files:**
- Create: `tests/test_q6_selected_profile_source_contract.py`
- Modify: `C:/models/expertflow/worktrees/llama-q6-selected-static/ggml/src/ggml-backend.cpp`
- Create: `docs/evidence/q6-selected-static/profile.json`
- Create: `docs/evidence/q6-selected-static/placement-manifest.json`

**Interfaces:**
- Consumes: fair stock `-ngl 99 --cpu-moe` Q6 configuration.
- Produces: disabled-by-default `LLAMA_EXPERTFLOW_SPLIT_PROFILE` JSONL aggregates with split backend, nodes, decode calls, input-boundary time, compute time, and explicit diagnostic synchronization status.

- [ ] Write source-contract tests requiring default-off behavior, bounded storage, deferred file output, and synchronization only while profiling.
- [ ] Run the focused test and verify it fails because the probe is absent.
- [ ] Implement the minimal fixed-capacity scheduler probe and deferred JSONL writer.
- [ ] Build CUDA Release and run the focused and complete ExpertFlow tests.
- [ ] Run one bounded stock diagnostic profile, reconcile selected layers and adjacent CUDA boundaries, and calculate the maximum plausible selected-layer contribution.
- [ ] Stop with `Q6 NO BOTTLENECK` if the measured contribution cannot plausibly exceed 5%; otherwise commit instrumentation with `Assisted-by: OpenAI Codex`.

### Task 2: Static Q6 proof

**Files:**
- Modify only if required by Q6 source-contract failure: `C:/models/expertflow/worktrees/llama-q6-selected-static/ggml/src/ggml-backend.cpp`
- Modify only if required by Q6 source-contract failure: `C:/models/expertflow/worktrees/llama-q6-selected-static/src/llama-context.cpp`
- Create: `docs/evidence/q6-selected-static/static-proof.json`

**Interfaces:**
- Consumes: existing `LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER=0,1,15,20` persistent-shadow path.
- Produces: exact Q6 bundle, arena, ownership, teardown, determinism, and hidden-duplicate evidence.

- [ ] Add a failing Q6 bundle/stride/alignment contract test only if the existing path does not already satisfy the contract.
- [ ] Run a short four-layer Q6 proof with CUDA graphs disabled and verify exact bundle bytes, stable repetitions, clean teardown, and no progressive memory growth.
- [ ] Stop with `Q6 STATIC STOP` on any forbidden architecture requirement; otherwise commit any necessary minimal runtime change separately.

### Task 3: Ten matched cold-process pairs

**Files:**
- Create: `scripts/run_q6_selected_static.py`
- Create: `tests/test_q6_selected_static_runner.py`
- Create: `docs/evidence/q6-selected-static/run-pairs.csv`
- Create: `docs/evidence/q6-selected-static/results.json`

**Interfaces:**
- Consumes: identical OFF/ON Q6 commands and the frozen alternating pair order.
- Produces: ten 512-token cold-process pairs with TPS, TTFT, latency, VRAM, environment, response hashes, and paired bootstrap interval.

- [ ] Write runner/parser/statistics tests and verify they fail before implementation.
- [ ] Implement the minimal hidden sequential runner with immutable per-run manifests and no ordinary-run exclusion.
- [ ] Run ten alternating matched pairs and compute paired mean, median, standard deviation, and 95% confidence interval.
- [ ] Stop with `Q6 PERFORMANCE STOP` if improvement is below 5%, the interval includes zero, or stability fails.

### Task 4: Frozen Q6 quality gates

**Files:**
- Create: `docs/evidence/q6-selected-static/quality-results.json`

**Interfaces:**
- Consumes: existing Q1b PPL and MMLU harnesses, unchanged corpora, seeds, and protocols.
- Produces: OFF/ON NLL, PPL, paired-bootstrap interval, MMLU score, and determinism evidence.

- [ ] Run frozen held-out PPL OFF/ON and enforce both `<= +1.0%` gates.
- [ ] Run frozen 100-item MMLU OFF/ON and deterministic repeats for disagreements or boundary results.
- [ ] Stop with `Q6 PERFORMANCE STOP` if either quality gate fails.

### Task 5: Final evidence and verification

**Files:**
- Create: `docs/evidence/q6-selected-static/report.md`
- Update: `docs/evidence/q6-selected-static/results.json`
- Update: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: all measured Stage 1-4 artifacts.
- Produces: exactly one declared verdict and bounded next recommendation.

- [ ] Separate measured facts, diagnostic synchronized timings, calculations, and limitations.
- [ ] Run all ExpertFlow tests with the selected llama source path and verify both worktrees are clean after commits.
- [ ] Commit final evidence separately; keep branches unmerged and unpushed.
