# Q6 Placement Optimizer Final Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and validate the fastest bounded Q6 full-static expert placement, stopping unless it reaches 23.5 TPS and beats the strongest fair stock runtime.

**Architecture:** Reuse the disabled synchronized split profiler for offline ranking, increase only the fixed static-island capacity, and drive a sequential two-layer greedy search with cold-process matched measurements. Run cleanup, tuning, final quality, and product packaging only after their explicit performance gates pass.

**Tech Stack:** Python 3.12, pytest, PowerShell, C++17, llama.cpp/ggml, MSVC v143, CUDA 12.8, CMake/Ninja.

## Global Constraints

- Q6 GGUF SHA-256 must remain `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.
- Sequential GPU processes only; no new kernels, scheduler redesign, whole-layer placement, downloads, Q4 work, or prediction before reactive caching beats static.
- Final quality limits: PPL point and 95% upper bound at most +1%; MMLU decline at most 1 percentage point.
- Preserve at least 256 MiB VRAM margin; keep changes isolated, unmerged, and unpushed.

---

### Task 1: Repeated layer profile and ranking

**Files:**
- Create: `scripts/analyze_q6_layer_profile.py`
- Create: `tests/test_q6_layer_profile.py`
- Create: `docs/evidence/q6-placement-final/layer-profile.json`
- Create: `docs/evidence/q6-placement-final/search-ledger.json`

**Interfaces:**
- Consumes one or more split-profile JSON files with `records`.
- Produces `rank_profiles(paths, selected_layers, shadow_bytes)` with per-layer timing medians and a deterministic descending `score_us_per_mib` ranking.

- [ ] Write a failing unit test using two synthetic profiles and assert median timing, selected flags, constant 685,933,056-byte cost, and descending ranking.
- [ ] Run `python -m pytest tests/test_q6_layer_profile.py -q` and confirm the missing analyzer fails.
- [ ] Implement the minimal parser/ranker and CLI JSON writer.
- [ ] Run the focused test, then collect three sequential 128-token stock profiles with `LLAMA_EXPERTFLOW_SPLIT_PROFILE` and no static islands.
- [ ] Write the measured layer profile and initialize the append-only search ledger; commit profiling/ranking support.

### Task 2: Bounded twelve-layer static capacity

**Files:**
- Modify: `tests/test_q1b_multi_island_source_contract.py`
- Modify: `C:\models\expertflow\worktrees\llama-q6-placement-final\src\llama-context.cpp`
- Modify: `C:\models\expertflow\worktrees\llama-q6-placement-final\ggml\src\ggml-backend.cpp`

**Interfaces:**
- `LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER` accepts at most twelve unique layer IDs.
- Each layer owns four bounded shadow entries, giving forty-eight maximum shadows.

- [ ] Change the source-contract test to require twelve layers and forty-eight shadows; run it against the llama worktree and confirm failure on the four-layer constants.
- [ ] Change only the two fixed constants in llama-context and ggml-backend.
- [ ] Run Q1b/Q6 source contracts and rebuild Release CUDA CLI/server/perplexity binaries.
- [ ] Verify feature-off behavior and a four-layer proof remain stable; commit the capacity implementation in llama.cpp and the updated contract in ExpertFlow.

### Task 3: Greedy two-layer static search

**Files:**
- Modify: `scripts/run_q6_selected_static_pairs.ps1`
- Modify: `tests/test_q6_selected_static_cli_runner.py`
- Update: `docs/evidence/q6-placement-final/search-ledger.json`

**Interfaces:**
- Harness accepts `-StaticLayers` and `-CudaGraphsDisabled` while preserving the frozen prompt/model/runtime contract.
- Each candidate contributes two complete alternating cold-process pairs and a candidate summary entry.

- [ ] Add failing harness-contract tests for explicit layer-set and graph-control parameters.
- [ ] Implement the two parameters without changing defaults used by earlier evidence.
- [ ] Test ranked additions from `[0,1,15,20]`, retain only candidates adding at least 0.25 TPS, and stop at 23.5 TPS or two consecutive failures.
- [ ] Record exact VRAM, deterministic hashes, commands, bundle allocation, and rejection reasons; commit only a winning static set milestone if one exists.

### Task 4: Gated cleanup and runtime tuning

**Files:**
- Modify only existing static-path source and its focused source contracts if provisional TPS reaches 23.5.
- Update: `docs/evidence/q6-placement-final/search-ledger.json`

**Interfaces:**
- Every candidate changes one bounded factor and is retained only on positive two-pair mean decode improvement.

- [ ] If below 23.5, skip hot-path cleanup and tune only enough to establish the fastest stable stock/static frontier.
- [ ] If at or above 23.5, inspect the static path for per-token work, add a failing source/behavior contract before each removal, and retain only measured improvements.
- [ ] Test CUDA graphs, then bounded batch, ubatch, thread, batch-thread, and supported flash-attention changes one variable at a time; record and revert non-winners.
- [ ] Stop dynamic stages unless static approaches/beats stock and VRAM alone blocks additional ranked layers.

### Task 5: Authoritative validation and terminal evidence

**Files:**
- Create: `docs/evidence/q6-placement-final/report.md`
- Create: `docs/evidence/q6-placement-final/results.json`
- Create: `docs/evidence/q6-placement-final/run-pairs.csv`
- Create: `docs/evidence/q6-placement-final/placement-manifest.json`
- Create: `docs/evidence/q6-placement-final/quality-results.json`
- Create: `docs/evidence/q6-placement-final/final-scorecard.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Final evidence names one terminal verdict and distinguishes diagnostic, smoke, authoritative, projected, and quarantined results.

- [ ] If provisional ExpertFlow reaches 23.5 and beats tuned stock, run ten alternating cold-process pairs plus one streaming latency pair; otherwise stop after sufficient static frontier evidence.
- [ ] Run held-out PPL and frozen MMLU only for a qualifying final winner; reuse already-frozen baseline evidence only when runtime/model/corpus protocol is unchanged and disclose it.
- [ ] Validate all JSON/CSV artifacts, run the full CPU suite and focused Q1b/Q6 source contracts, check process cleanup and `git diff --check`.
- [ ] Commit final evidence separately, keep both branches clean/unmerged/unpushed, and issue exactly the declared verdict.
