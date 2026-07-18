# Q6 Quality-Preserving Expert Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove or reject the existing layer-0 CUDA expert island under frozen quality gates before implementing any reactive streaming.

**Architecture:** ExpertFlow owns immutable benchmark manifests, analysis, and evidence; an isolated llama.cpp worktree owns the disabled-by-default static CUDA shadow. The first executable milestone is Q1 only: feature-off and feature-on builds share one binary, model, tokenizer, inputs, and settings, with persistent complete Q4 expert bundles and no eviction.

**Tech Stack:** Python 3.12, pytest, stdlib JSON/hash/process utilities, llama.cpp at `a7312ae94f801fc9c6786dc56e38df57b964f697`, CMake/Ninja, MSVC v143 14.39, CUDA 12.8.

## Global Constraints

- Preserve `codex/q6-runtime-final` at `378ce57` as the exactness rollback floor.
- Work only in `codex/q6-quality-preserving` and `codex/q6-quality-preserving-llama`; do not merge or push.
- Keep the runtime feature disabled by default.
- Q1 changes only static layer-0 placement; do not implement eviction, prediction, scheduler replacement, a new GGML operation, or a replacement CUDA kernel.
- Stop before Q2 if relative perplexity increase exceeds 0.5%, MMLU accuracy falls by more than 1.0 percentage point, determinism fails, long-generation gates fail, or memory safety/teardown fails.
- Keep commands, durations, hashes, results, and failures append-only in `docs/evidence/q6-quality-preserving/`.

---

### Task 1: Frozen Quality Manifest and Result Analysis

**Files:**
- Create: `src/expertflow/quality/manifest.py`
- Create: `src/expertflow/quality/analysis.py`
- Create: `src/expertflow/quality/__init__.py`
- Create: `scripts/freeze_quality_manifest.py`
- Test: `tests/test_quality_manifest.py`
- Test: `tests/test_quality_analysis.py`

**Interfaces:**
- Produces: `freeze_manifest(config: FreezeConfig) -> dict[str, object]`, `sha256_file(path: Path) -> str`, and `evaluate_quality_gate(reference: dict, candidate: dict) -> dict`.

- [ ] **Step 1: Write failing manifest tests**

```python
def test_freeze_manifest_is_stable_and_content_addressed(tmp_path):
    manifest = freeze_manifest(_fixture_config(tmp_path))
    assert manifest["schema_version"] == 1
    assert len(manifest["mmlu"]["items"]) == 100
    assert manifest["mmlu"]["selection_salt"] == "expertflow-option1-v1"
    assert manifest["wikitext"]["token_count"] == 8192
    assert manifest["manifest_sha256"] == canonical_manifest_hash(manifest)
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run: `uv run pytest tests/test_quality_manifest.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'expertflow.quality'`.

- [ ] **Step 3: Implement deterministic freezing**

Implement `FreezeConfig` with dataset revisions, local source files, tokenizer command, model/executable paths, sampler settings, and output path. Select each MMLU subject's ten rows by ascending `sha256(f"expertflow-option1-v1:{subject}:{row_id}")`; reject duplicate IDs, mutable revisions such as `main`, non-100 item totals, and non-8192 WikiText token totals. Hash every input and emit sorted, indented JSON via an atomic temporary-file replacement.

- [ ] **Step 4: Write failing gate-analysis tests**

```python
def test_quality_gate_applies_frozen_thresholds():
    result = evaluate_quality_gate(
        {"perplexity": 10.0, "mmlu_correct": 80, "mmlu_total": 100,
         "repeated_4gram_rate": .10, "distinct_2": .70},
        {"perplexity": 10.04, "mmlu_correct": 79, "mmlu_total": 100,
         "repeated_4gram_rate": .14, "distinct_2": .66},
    )
    assert result["pass"] is True
    assert result["perplexity_relative_change"] == pytest.approx(.004)
```

- [ ] **Step 5: Implement and verify analysis**

`evaluate_quality_gate` must calculate relative perplexity, percentage-point accuracy/degeneration deltas, candidate self-determinism, process exit status, and explicit named pass/fail reasons. Run `uv run pytest tests/test_quality_manifest.py tests/test_quality_analysis.py -q`; expect all tests to pass.

- [ ] **Step 6: Commit the tooling**

Run: `git add src/expertflow/quality scripts/freeze_quality_manifest.py tests/test_quality_manifest.py tests/test_quality_analysis.py && git commit -m "feat: freeze Q6 quality evaluation contract"`.

### Task 2: Immutable Benchmark Inputs

**Files:**
- Create: `docs/evidence/q6-quality-preserving/quality-manifest.json`
- Create: `docs/evidence/q6-quality-preserving/command-ledger.md`
- Create: `docs/evidence/q6-quality-preserving/README.md`
- Test: `tests/test_quality_evidence.py`

**Interfaces:**
- Consumes: `freeze_manifest` from Task 1.
- Produces: immutable model, executable, dataset, prompt, selection, and command identity used by every Q1 run.

- [ ] **Step 1: Resolve immutable dataset revisions and download only required files**

Use Hugging Face repository metadata to resolve `Salesforce/wikitext` and `cais/mmlu` to 40-character commit revisions. Record the exact commands, durations, resolved revisions, file sizes, and SHA-256 hashes in the ledger. Store downloaded datasets outside git under `C:\models\expertflow\quality-data\<revision>`.

- [ ] **Step 2: Write the failing evidence-contract test**

```python
def test_quality_manifest_is_frozen_and_complete():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(data["datasets"]["wikitext"]["revision"]) == 40
    assert len(data["datasets"]["mmlu"]["revision"]) == 40
    assert data["mmlu"]["item_count"] == 100
    assert data["wikitext"]["token_count"] == 8192
    assert data["thresholds"]["perplexity_relative_max"] == 0.005
```

- [ ] **Step 3: Freeze the manifest once**

Run `uv run python scripts/freeze_quality_manifest.py --config docs/evidence/q6-quality-preserving/freeze-config.json`. The command must refuse to overwrite an existing manifest unless its canonical hash is identical.

- [ ] **Step 4: Verify and commit inputs**

Run `uv run pytest tests/test_quality_evidence.py -q` and `git diff --check`; expect PASS. Commit only manifests, hashes, prompts, tests, and the ledger; do not commit dataset payloads or model files.

### Task 3: Static CUDA Island Source Contract

**Files:**
- Create: `tests/test_q6_quality_static_source_contract.py`
- Create in llama worktree: `C:\models\expertflow\worktrees\llama-q6-quality-preserving\src\llama-context.cpp` modifications
- Create in llama worktree: `C:\models\expertflow\worktrees\llama-q6-quality-preserving\ggml\src\ggml-backend.cpp` modifications

**Interfaces:**
- Produces: environment flag `LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER`; absent or negative means disabled, non-negative selects exactly one layer.

- [ ] **Step 1: Create the isolated llama.cpp worktree**

From `C:\models\expertflow\dependencies\llama.cpp-a7312ae-git`, create branch `codex/q6-quality-preserving-llama` at exact commit `a7312ae94f801fc9c6786dc56e38df57b964f697` and worktree `C:\models\expertflow\worktrees\llama-q6-quality-preserving`. Record repository status and commit hash.

- [ ] **Step 2: Write failing source-contract tests**

The tests must assert: the flag defaults disabled; the complete Q4 bundle consists of exactly `blk.<layer>.ffn_gate_up_exps.weight`, `blk.<layer>.ffn_down_exps.weight`, and `blk.<layer>.ffn_down_exps.scale`; shadow buffers are scheduler-owned and freed; `ffn_moe_weighted`/`ffn_moe_out` remain on the original backend; scale/view chains are not recursively reassigned; no diagnostic bypass flags remain.

- [ ] **Step 3: Run the source-contract tests and confirm failure**

Run: `$env:EXPERTFLOW_LLAMA_SOURCE='C:\models\expertflow\worktrees\llama-q6-quality-preserving'; uv run pytest tests/test_q6_quality_static_source_contract.py -q`

Expected: FAIL because the pristine source lacks `LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER`.

- [ ] **Step 4: Implement the minimal static island**

In `llama-context.cpp`, before scheduler assignment, route only the selected layer's `MUL_MAT_ID`, expert `MUL`, and `GLU` island to CUDA; recurse through `MUL` source 0 only, leaving scale/view source 1 untouched; explicitly keep combine/residual operations on their existing backend. In `ggml-backend.cpp`, add fixed-capacity scheduler-owned shadow records containing source tensor, persistent GGML context, backend buffer, shadow tensor, and loaded state. Allocate each complete expert-indexed tensor once on CUDA, copy it once, substitute it before graph splitting, and free buffers/contexts at scheduler teardown. Fail with a clear error on missing components, incompatible shapes/types, allocation failure, or capacity overflow.

- [ ] **Step 5: Verify source contracts and disabled build equivalence**

Run the focused test, then build `llama-cli` and `llama-perplexity`. Hash both binaries. Run the canonical short prompt with the flag absent and compare generated tokens and placement to the pristine matched build; require equality.

- [ ] **Step 6: Commit the llama milestone separately**

Commit only the two narrow llama.cpp source files with message `feat: restore disabled static CUDA expert island` on `codex/q6-quality-preserving-llama`.

### Task 4: Static Q4 Smoke, Memory, and Determinism

**Files:**
- Create: `scripts/run_q1_static_quality.py`
- Create: `tests/test_q1_static_quality_runner.py`
- Create: `docs/evidence/q6-quality-preserving/q1-smoke.json`

**Interfaces:**
- Produces: normalized per-run commands, exit codes, token IDs, router events, peak process VRAM/RAM, elapsed time, and teardown status.

- [ ] **Step 1: Write the failing runner test**

Use a fake executable that emits fixed llama timing lines and token/trace JSON. Assert the runner uses fresh processes, carries only the selected feature environment variable, samples process-owned memory, enforces timeouts, and records executable/model hashes.

- [ ] **Step 2: Implement the minimal runner and pass its unit tests**

Run `uv run pytest tests/test_q1_static_quality_runner.py -q`; expect PASS. The runner must never infer success from stdout alone: require exit code zero, complete output artifacts, no CUDA/NaN/Inf diagnostic, and child-process exit.

- [ ] **Step 3: Run feature-off and feature-on smoke**

Use the Q4 model and layer 0 at `-ngl 10`, fixed seed, greedy sampling, and the canonical prompt. Require complete persistent bundles, expected 408.375 MiB arena, stable process-owned memory, and clean teardown.

- [ ] **Step 4: Run three fresh candidate repetitions**

Require exact prompt tokens, generated tokens, and router events across all three feature-on runs. Compare feature-on to feature-off descriptively and preserve every divergence; cross-mode bit parity is not a Q1 requirement.

- [ ] **Step 5: Stop or commit**

If the smoke, memory, or candidate self-determinism gate fails, write a STOP report and do not run scored quality. Otherwise commit the runner, tests, and smoke evidence.

### Task 5: Frozen Q1 Quality Evaluation

**Files:**
- Create: `scripts/run_q1_quality_gate.py`
- Create: `tests/test_q1_quality_gate_runner.py`
- Create: `docs/evidence/q6-quality-preserving/q1-quality-results.json`
- Create: `docs/evidence/q6-quality-preserving/q1-quality-report.md`

**Interfaces:**
- Consumes: frozen manifest from Task 2 and matched binaries from Task 3.
- Produces: the terminal Q1 PASS/STOP decision.

- [ ] **Step 1: Write failing orchestration tests**

Test that feature-off always runs before feature-on; commands are derived from the immutable manifest; results with wrong dataset/model/executable hashes are rejected; partial runs cannot be scored; and every metric is labelled measured or calculated.

- [ ] **Step 2: Implement orchestration and pass unit tests**

Run `uv run pytest tests/test_q1_quality_gate_runner.py tests/test_quality_analysis.py -q`; expect PASS.

- [ ] **Step 3: Measure matched WikiText perplexity**

Run the four frozen 2048-token chunks with feature off, then feature on, using the same `llama-perplexity` binary and settings. Record each chunk and aggregate perplexity plus relative delta.

- [ ] **Step 4: Measure fixed 100-item MMLU accuracy**

Run the exact manifest order with greedy decoding and the frozen answer extractor. Preserve raw outputs, parsed answers, per-subject accuracy, total accuracy, and candidate-minus-reference percentage points.

- [ ] **Step 5: Measure long-generation diagnostics and routing overlap**

Run all six fixed 256-token prompts in fresh processes. Calculate repeated-4-gram, distinct-2, top-8 set overlap, ordered-position overlap, exact-set rate, and exact-order rate by layer and phase. These diagnostics cannot override a perplexity or MMLU failure.

- [ ] **Step 6: Apply the gate without retuning**

Generate `q1-quality-results.json` and `q1-quality-report.md` from the frozen thresholds. If any gate fails, declare `OPTION 1 Q1 STOP`, preserve evidence, and switch product work to the already-protected ExpertFlow Deploy fallback. If all gates pass, declare `OPTION 1 Q1 PASS` and authorize a separate Q2 design; do not implement Q2 in this task.

- [ ] **Step 7: Run full verification and commit evidence**

Run `uv run pytest -q`, `git diff --check`, verify both worktrees are otherwise clean, then commit the runner, tests, manifests, report, and append-only ledger. Keep both branches unmerged and unpushed.
