# Gate 3 Cross-Runtime Divergence Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify the Gate 3 cross-binary routing/token divergence as benign build-dependent numerical drift or evidence of an incorrect clean runtime, without changing the canonical llama.cpp source or creating cache code.

**Architecture:** Preserve `1fc549bd87ed58aea7c603c59ac26037324ec157` and the protected Observatory, derive a complete configuration manifest from durable artifacts and source defaults, then use one diagnostic-only probe binary against copied reference and clean runtime DLL sets. Capture one pre-top-k tensor at a time to minimize callback-induced scheduling changes, compare the first divergent selection numerically, and test the clean runtime repeatedly before applying the written decision rule.

**Tech Stack:** PowerShell, Python 3.12, JSON/JSONL, C++17 diagnostic probe, MinGW UCRT g++, CMake/Ninja, llama.cpp `a7312ae94f801fc9c6786dc56e38df57b964f697`, official CUDA 12.4 b10002 runtime, local MSVC 19.39/CUDA 12.8.93 Release runtime, pytest.

## Global Constraints

- Time-box the audit to 90 minutes of active investigation; stop earlier when the decision rule is satisfied.
- Keep `C:\sem4\Expertflow` and the annotated Observatory tag untouched.
- Keep `C:\models\expertflow\dependencies\llama.cpp-a7312ae-git` byte-clean.
- Preserve commit `1fc549bd87ed58aea7c603c59ac26037324ec157` and all existing Gate 3 artifacts.
- Do not create cache, slot-planner, transfer-path, predictor, asynchronous-stream, MTP, or allocator code during the audit.
- Treat the official/prebuilt runtime comparison as diagnostic only until the decision rule is applied.
- Log every command, configuration, failure, measurement, artifact, and decision under `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit`.

---

### Task 1: Preserve and inventory the audit boundary

**Files:**
- Create externally: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit\commands.jsonl`
- Create externally: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit\preflight.json`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: commit `1fc549b`, protected commit `d846bdf`, clean llama.cpp commit `a7312ae`.
- Produces: immutable audit provenance and a clean baseline.

- [x] **Step 1: Record worktree, protected checkout, source checkout, GPU, environment, and process state**

Run exact Git identity/status commands for all three checkouts, `nvidia-smi`, relevant-process inspection, and `EXPERTFLOW_LIVE_CACHE*` enumeration. Hash both probe binaries, both runtime DLL sets, the model, and the attachment text.

- [x] **Step 2: Verify the repository baseline**

Run `uv run pytest -q` and require 87 passing tests. Abort audit source changes if this fails.

### Task 2: Build the complete comparison manifest

**Files:**
- Create externally: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit\comparison-manifest.json`

**Interfaces:**
- Consumes: the reference and clean Gate 3 manifests, stderr logs, probe source, local `CMakeCache.txt`, official b10002 asset metadata, and pinned release workflow.
- Produces: explicit `equal`, `different`, or `not_recoverable_from_release_artifact` classifications for every requested field.

- [x] **Step 1: Reconcile model, prompt, tokenizer, and decode inputs**

Record GGUF path/hash, exact UTF-8 prompt bytes/hash, prompt token IDs, raw-prompt/no-chat-template behavior, greedy sampler chain, context request and padded context, GPU layers, thread counts, seed applicability, tensor split, split mode, device, flash attention, KV types, mmap/mlock, batch, and ubatch.

- [x] **Step 2: Reconcile builds and backend selection**

Record source/build IDs, probe hashes, compiler/linker provenance, CUDA toolkit/runtime, driver, CUDA architecture, Release flags, CMake flags, loaded CPU/CUDA backend filenames, and every runtime DLL hash. Do not infer unavailable official flags; label them explicitly as unrecoverable and cite the release workflow where possible.

### Task 3: Capture the first divergent router boundary

**Files:**
- Modify: `native/router_probe/main.cpp`
- Create externally: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit\diagnostic-runtime-reference`
- Create externally: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit\diagnostic-runtime-clean`
- Create externally: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit\score-comparison.json`

**Interfaces:**
- Consumes: an exact tensor name, global token index, and output path.
- Produces: unchanged default probe behavior plus optional full contiguous F32/I32 diagnostic tensor serialization.

- [x] **Step 1: Add an opt-in single-tensor diagnostic interface**

Add `--capture-tensor NAME`, `--capture-token-index INDEX`, and `--capture-output FILE`. When all three are present, the callback requests only the exact named tensor for the target active forward in addition to the existing `ffn_moe_topk-*` trace. Serialize tensor name/type/dimensions, global token index, raw numeric values, and byte count. With no capture options, preserve current behavior byte-for-byte at the JSON schema level.

- [x] **Step 2: Build one probe and create isolated runtime copies**

Build the diagnostic probe once against the pinned ABI. Copy the same probe executable, verified by SHA-256, beside copied official-reference DLLs and copied clean DLLs. Never overwrite either original runtime.

- [x] **Step 3: Capture token index 0/token ID 2, layer 24 one boundary at a time**

Run separate identical commands for `ffn_moe_argsort-24`, `ffn_moe_logits-24`, and the nearest already-named layer-input tensor. Use the frozen prompt, `-n 1`, `-ngl 10`, and 12 threads. Preserve selected top-k traces for each capture run.

- [x] **Step 4: Compute the numerical explanation**

Report all 128 pre-top-k scores, sorted top-M IDs/scores, selected IDs and weights where available, rank-8/rank-9 gap in each runtime, maximum absolute/relative cross-runtime difference, entering/leaving experts, tensor checksums, and whether the swap margin is comparable to observed drift. If the diagnostic callback changes the selected set, report that and use host ranking of captured pre-top-k values rather than hiding the perturbation.

### Task 4: Test clean-runtime stability

**Files:**
- Create externally: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit\stability-summary.json`

**Interfaces:**
- Consumes: three frozen representative prompts and the exact clean Release runtime.
- Produces: per-prompt trace-off/on and three-repeat token/router/memory results.

- [x] **Step 1: Run representative deterministic repetitions**

Use the baseline general prompt plus one code and one translation prompt from the frozen public corpus. For each, run trace off once and trace on three times with greedy sampling, 16 generated tokens, ten GPU layers, one-token batches, and 12 threads.

- [x] **Step 2: Validate per-prompt stability (failed required off/on parity for code and translation)**

Require exact tokens across all four runs, exact ordered router selections across trace repeats, complete 30-layer/eight-expert events, strict causal order, no persistent process, and no settled GPU/host allocation growth.

### Task 5: Run a controlled local alternative comparison

**Files:**
- Create externally: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit\alternate-build\release-vs-debug-router.json`

**Interfaces:**
- Consumes: the existing exact unmodified local Debug or a single controlled alternate local build.
- Produces: evidence whether compiler/backend selection changes the same pre-top-k boundary.

- [x] **Step 1: Prefer an existing exact local Debug runtime**

Inventory the existing Debug build. If it contains loadable inference DLLs, run the same single-tensor token-2/layer-24 capture on CPU and compare it with local Release CPU. If it cannot run inference, record that exact limitation and perform one bounded local Release CPU comparison instead of starting a broad rebuild.

- [x] **Step 2: Attribute backend/build differences conservatively**

Separate CPU backend dispatch differences, CUDA toolkit/code-generation differences, and compiler optimization differences. Do not claim one cause unless a one-variable comparison isolates it.

### Task 6: Apply the decision rule and checkpoint

**Files:**
- Create: `docs/evidence/live-cache/gate3-divergence-audit.md`
- Modify: `configs/llama-a7312ae-cuda128.json`
- Modify: `docs/evidence/live-cache/gate3-clean-llama.md`
- Modify: `docs/superpowers/plans/2026-07-15-bounded-live-cache.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: comparison manifest, score comparison, stability summary, alternative comparison, and all raw artifacts.
- Produces: `benign-build-drift` or `blocked-likely-defect`, with no ambiguous pass state.

- [x] **Step 1: Classify the divergence (cross-runtime drift benign; Gate 3 blocked on clean trace safety)**

Classify as benign only if inputs/options are equivalent, clean Release is deterministic and trace-safe, the first set change is explained by near-boundary numerical drift, no malformed tensor/token/layer/memory evidence exists, generated output is structurally valid, and existing tests pass. Otherwise keep Gate 3 blocked and name the failed condition.

- [x] **Step 2: Verify and commit the audit**

Run 87 ExpertFlow tests, replay reconciliation, artifact/hash reconciliation, JSON/JSONL parsing, `git diff --check`, protected/source cleanliness, and no-cache checks. Commit only reproducible helper/evidence/configuration files; never commit runtime binaries or generated score arrays.

### Task 7: Conditional canonical-runtime handoff

**Files:**
- Create only after a benign classification: `docs/evidence/live-cache/gate3-canonical-runtime.md`
- Create only after a benign classification: `docs/evidence/live-cache/gate4-one-layer-recommendation.md`

**Interfaces:**
- Consumes: a committed benign audit.
- Produces: exact canonical binary hashes and a bounded recommendation; it does not itself implement cache code.

- [ ] **Step 1: Establish the canonical baseline**

Hash the exact unmodified local Release runtime, repeat the canonical cache-off runs, and state that future correctness compares cache-disabled versus cache-enabled within that same build, model, prompt, and decode configuration. Keep cross-binary parity diagnostic only.

- [ ] **Step 2: Issue the separate one-layer recommendation**

Name intended llama.cpp files, disabled-by-default feature flag, exact parity tests, rollback commit, expected one/two-slot memory behavior, forced-swap tests, and stop conditions. Only then resume the existing bounded Gate 4 plan.
