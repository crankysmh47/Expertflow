# ExpertFlow Bounded Physical-Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether a minimal exact blocking live-cache spike is physically justified, using a 40-conversation Q4 corpus, exhaustive expert-byte accounting, independent CUDA transfer measurements, and held-out deadline estimates without modifying the live llama.cpp cache path.

**Architecture:** Keep raw prompts, paired probe outputs, runtime logs, and generated measurement JSON under `C:\models\expertflow`; check in only reproducible manifests, analysis code, tests, evidence documents, and bounded fixtures. Fit static residents on the frozen training split, evaluate validation/test conversations independently, and keep Vulkan callback windows, CUDA transfer measurements, cross-backend simulation, oracle estimates, and any future live-runtime measurements as separate evidence classes.

**Tech Stack:** Python 3.11, dependency-free ExpertFlow CLI, pytest, pinned llama.cpp b10002 Vulkan router probe, pinned llama.cpp b10002 CUDA runtime libraries, GGUF metadata reader from the pinned llama.cpp source archive, and NVIDIA CUDA Runtime API through `ctypes`.

## Global Constraints

- Use the verified `google/gemma-4-26B-A4B-it` Q4_0 GGUF and pinned llama.cpp b10002 revision `a7312ae94f801fc9c6786dc56e38df57b964f697`.
- Keep all large artifacts on `C:\`; do not write model or run artifacts to `D:\`.
- Collect 40 independent conversations: 32 train, 4 validation, and 4 test; never split events from one conversation across sets.
- Use deterministic greedy decoding, 64 generated tokens where feasible, 10 GPU-offloaded layers, 12 CPU threads, and paired tracing-disabled/tracing-enabled runs.
- Keep `live_cache_enabled=false`; do not make runtime speedup, KV-cache, CUDA deadline, or copy/compute-overlap claims from simulator evidence.
- Do not modify the live llama.cpp expert cache path in this plan.
- Append decisions, exact commands, durations, configurations, failures, artifacts, hashes, and commits to `PROJECT_LOG.md`.

---

### Task 1: Frozen 40-conversation corpus and resumable paired collector

**Files:**
- Create: `configs/q4-physical-feasibility-corpus.json`
- Create: `src/expertflow/collection.py`
- Create: `tests/test_collection.py`
- Modify: `src/expertflow/cli/main.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: a JSON corpus with `conversation_id`, `split`, `domain`, `source`, `messages`, and `prompt`.
- Produces: `collect_trace_pairs(corpus_path: Path, config: CollectionConfig) -> dict[str, object]` and `expertflow collect-pairs` with a resumable `collection-manifest.json`.

- [x] Write a failing test proving conversation IDs are unique, the split is exactly 32/4/4, all eight declared domains are covered, and malformed manifests fail before launching a process.
- [x] Run `uv run pytest tests/test_collection.py -q` and confirm the failure is caused by the absent collection module.
- [x] Implement the smallest manifest validator and process runner. Store argv arrays, runtime/model/probe hashes, start/end timestamps, duration, native exit code, output hashes, and parity result per conversation.
- [x] Add resume validation that skips a pair only when both token files, the trace, logs, hashes, and exact parity report are present and valid.
- [x] Run the focused tests and the full suite, then commit the collector separately.
- [x] Launch all 40 pairs under `C:\models\expertflow\runs\physical-feasibility-q4-vulkan`, reporting progress after each bounded batch.

### Task 2: Train-only static fitting and prompt/domain held-out breakdown

**Files:**
- Create: `src/expertflow/analysis/heldout_breakdown.py`
- Create: `tests/test_heldout_breakdown.py`
- Modify: `src/expertflow/cli/main.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: the frozen collection manifest, training traces, and validation/test traces.
- Produces: `build_heldout_breakdown(...) -> dict[str, object]` with per-prompt, per-domain, per-split, phase-separated static-96 and reset-per-conversation LRU metrics.

- [x] Write a failing test showing that static residents come only from training events and that an evaluation conversation cannot affect any later conversation's initial cache.
- [x] Verify the focused red failure, then implement the minimum evaluator by reusing the current policy outcome engine.
- [x] Add weighted aggregate reconciliation tests: prompt totals must sum to domain, split, and global totals.
- [x] Generate prefill and decode breakdown artifacts and record every prompt's parity, event count, demand count, hit rate, cold bytes/token, and serialized transfer estimate.
- [x] Commit the evaluator and evidence checkpoint.

### Task 3: Exhaustive packed expert inventory and static-96 physical fit

**Files:**
- Create: `src/expertflow/analysis/expert_layout.py`
- Create: `scripts/measure_q4_expert_layout.py`
- Create: `tests/test_expert_layout.py`
- Create: `docs/evidence/q4-static-96.md`
- Create: `docs/evidence/q4-expert-layout.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: tensor name, dimensions, quantization type, encoded bytes, GGUF offset/alignment, layer ID, and expert count from the pinned GGUF reader.
- Produces: `build_expert_inventory(tensors, *, alignment: int) -> dict[str, object]` and a JSON row for every one of 30 layers x 128 experts.

- [x] Write a failing synthetic-layout test covering fused gate/up, down projection, per-expert scale, tensor-end padding, and conservative slot alignment.
- [x] Verify the red failure, implement byte-span accounting, and reject non-divisible expert tensors or overlapping/out-of-bounds spans.
- [x] Run the script over the real Q4 file and independently reconcile every layer-expert object, every layer total, all 3,840 objects, and the projected 21 x 96 slot allocation.
- [x] Document that `96` means 96 resident expert slots per target layer, not 96 global slots; state exact encoded/aligned bytes, target layers, selection procedure, phase, fit scope, and why this differs from the earlier prompt-local static-8 36.37% result.
- [x] Commit the inventory and documentation checkpoint.

### Task 4: Independent CUDA transfer latency and enqueue benchmark

**Files:**
- Modify: `src/expertflow/runtime/cuda_transfer.py`
- Modify: `tests/test_cuda_transfer.py`
- Modify: `src/expertflow/cli/main.py`
- Modify: `docs/evidence/q4-expert-transfer.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: `cudart` DLL, exact payload sizes, batch/copy counts, warmups, single-copy sample count, and device index.
- Produces: pageable-to-pinned staging, pageable H2D, pinned H2D, single-copy CUDA-event p50/p95 latency, host enqueue p50/p95, and sustained batch bandwidth.

- [x] Add a failing test for `measure_single_copy` output and `cudaMemcpyAsync` host-enqueue overhead, with explicit units and sample provenance.
- [x] Verify the red failure, then implement the narrow CUDA Runtime calls without adding a toolkit or framework dependency.
- [x] Run idle-GPU trials for 1 byte, the exact component sizes, exact packed expert size, aligned slot size, and eight-slot layer fill; preserve raw samples.
- [x] Repeat enough independent trials to report p50/p95 and note WDDM, default stream, idle GPU, no concurrent model, and no copy/compute contention.
- [x] Commit code and measured evidence.

### Task 5: Measured transfer sensitivity in the deadline simulator

**Files:**
- Modify: `src/expertflow/analysis/deadline.py`
- Modify: `tests/test_deadline.py`
- Modify: `src/expertflow/cli/main.py`
- Create: `docs/evidence/q4-deadline-sensitivity.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: frozen training traces, validation/test decode traces, static capacity, measured CUDA transfer p50/p95, and observed Vulkan callback windows.
- Produces: a scenario matrix whose fields name transfer backend, window backend, statistic, and evidence kind.

- [x] Write a failing test proving the simulator cannot serialize a mixed-backend result without both backend labels and an `estimated_cross_backend` classification.
- [x] Implement p50/p95 transfer scenarios while preserving the existing one-layer perfect-future oracle as a separate non-deployable upper bound.
- [x] Rerun validation and test scenarios, and reconcile no-prefetch, oracle, static, and LRU totals against the held-out breakdown.
- [x] Commit the simulator/evidence checkpoint without changing the live-cache verdict.

### Task 6: Continuously shippable Observatory and visual replay verification

**Files:**
- Modify: `src/expertflow/reporting.py`
- Modify: `tests/test_reporting.py`
- Modify: `README.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: static-96 contract, exhaustive byte inventory, per-prompt/domain results, transfer measurements, deadline scenarios, and gate verdict.
- Produces: a standalone report that keeps evidence classes visibly separate and a clean judge replay path.

- [ ] Add failing report tests for per-domain tables, the static-96 definition, backend/evidence labels, and `live_cache_enabled=false`.
- [ ] Implement the minimum renderer changes and regenerate the external HTML.
- [ ] Serve it over localhost, inspect the complete page visually, capture the server command and result, and fix any clipping or misleading labels.
- [ ] Reproduce the checked-in judge fixture from a clean C-drive archive/setup and record exact commands and duration.
- [ ] Run all tests, Python compilation, CLI help, TOML parsing, local-link validation, artifact reconciliation, and `git diff --check`; commit the shippable checkpoint.

### Task 7: Written go/no-go for a minimal exact blocking cache spike

**Files:**
- Create: `docs/evidence/q4-live-cache-go-no-go.md`
- Modify: `README.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: only the verified outputs of Tasks 1-6.
- Produces: `PROCEED`, `CONDITIONAL`, or `NO-GO`, with explicit gates for fit, measured transfer headroom, non-oracle held-out policy, and feature-flag isolation.

- [ ] State each gate with direct artifact paths/hashes and separate measured facts from estimates.
- [ ] Recommend a live spike only if all four user-defined conditions pass; otherwise keep the Observatory-first direction.
- [ ] If approved, write a separate plan for the exact blocking, feature-flagged slot experiment. Do not implement asynchronous prefetch, prediction, MTP, or speculative decoding.
- [ ] Run final verification and commit the decision memo.

## Stop Rules

- Stop before any live llama.cpp cache modification unless Task 7 explicitly returns `PROCEED` and the user approves the follow-on spike.
- A failed parity pair is retained and reported; it is never silently replaced.
- A missing or inconsistent artifact blocks resume for that pair and forces a rerun.
- CUDA transfer measurements do not become CUDA layer deadlines, and Vulkan callback windows do not become live runtime measurements.
- Static hit rate, cold bytes, and serialized transfer estimates do not become speedup claims.
