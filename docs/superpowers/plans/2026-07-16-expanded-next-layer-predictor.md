# Expanded Next-Layer Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, select, lock, and evaluate the bounded next-layer predictor on the frozen canonical 60/12/12 expanded corpus.

**Architecture:** Extend the existing leakage-safe dataset and predictor runner without changing the learned-model architectures. Add bounded B2 configuration and admission-policy support, select only on validation, write an immutable lock, then permit one guarded test evaluation.

**Tech Stack:** Python 3.11+, pytest, NumPy, CPU PyTorch, JSON/JSONL, pickle, SHA-256.

## Global Constraints

- Preserve exactly 60 train, 12 validation, and 12 sealed test conversations.
- Preserve six-domain 10/2/2 balance and unique conversation/prompt identities.
- Search only four B2 configurations, widths 8/12/16, and two admission rules.
- Do not tune B3/B4 or integrate with llama.cpp/runtime code.
- Open expanded test exactly once after an immutable selection lock exists.
- Label cache accounting simulated and make no speedup claim.

---

### Task 1: Expanded corpus contract

**Files:**
- Modify: `src/expertflow/predictor/dataset.py`
- Modify: `tests/test_predictor_dataset.py`

**Interfaces:**
- Consumes: canonical collection manifest and trace shards.
- Produces: `load_pilot_dataset(..., expected_split_counts, expected_domain_counts, require_unique_prompt_hashes)` with fail-closed provenance checks.

- [ ] Add failing tests for 60/12/12 counts, 10/2/2 domain balance, and duplicate prompt hashes.
- [ ] Run `python -m pytest tests/test_predictor_dataset.py -q` and confirm the new tests fail for missing arguments/validation.
- [ ] Add the minimal dataset validation while preserving existing pilot defaults.
- [ ] Re-run the dataset tests and confirm all pass.

### Task 2: Bounded B2 configurations

**Files:**
- Modify: `src/expertflow/predictor/models.py`
- Modify: `tests/test_predictor_models.py`

**Interfaces:**
- Consumes: `PredictionSample`.
- Produces: `TransitionPredictor.fit(samples, weighting, phase_mode)`, `scores(sample)`, and `has_support(sample, expert_id)`.

- [ ] Add failing tests showing source normalization changes ranking, phase-specific tables isolate prefill/decode, and support excludes deterministic fallback.
- [ ] Run `python -m pytest tests/test_predictor_models.py -q` and confirm failures are caused by the absent API.
- [ ] Implement only `raw_count|source_normalized` and `pooled|separate`.
- [ ] Re-run predictor-model tests and confirm all pass.

### Task 3: Domain metrics and admission-aware shadow

**Files:**
- Modify: `src/expertflow/predictor/metrics.py`
- Modify: `src/expertflow/predictor/shadow.py`
- Modify: `tests/test_predictor_metrics.py`
- Modify: `tests/test_predictor_shadow.py`

**Interfaces:**
- Produces: `per_domain` metrics and `simulate_shadow(..., admitted_rankings=...)` accounting for candidates rejected by admission.

- [ ] Add failing tests for per-domain output and observed-support admission accounting.
- [ ] Run the two focused test modules and confirm expected failures.
- [ ] Add the smallest grouping and admitted-prefix behavior.
- [ ] Re-run focused tests and confirm all pass.

### Task 4: Expanded fit/lock/test pipeline

**Files:**
- Modify: `scripts/run_next_layer_predictor.py`
- Modify: `tests/test_predictor_pipeline.py`

**Interfaces:**
- CLI: `fit|test --manifest PATH --output PATH --expanded`.
- Produces: validation metrics, artifacts, immutable selection lock, one test metrics file, ledger, and artifact index.

- [ ] Add failing pipeline tests for expanded contract, immutable lock fields, selected configuration, and refusal of a second test command.
- [ ] Run `python -m pytest tests/test_predictor_pipeline.py -q` and confirm expected failures.
- [ ] Implement expanded fit, fixed decision rule, lock hashes, and guarded test.
- [ ] Re-run pipeline tests and the complete test suite.

### Task 5: Validation freeze and one-time test

**Files:**
- Create externally: `C:\models\expertflow\runs\expanded-predictor-final\*`

**Interfaces:**
- Consumes: exact expanded manifest.
- Produces: immutable selection and sealed-test evidence.

- [ ] Run the expanded `fit` command and verify `selection-lock.json` says `test_opened=false`.
- [ ] Hash and independently parse every validation artifact and lock field.
- [ ] Run the expanded `test` command exactly once.
- [ ] Verify a second test command is rejected without modifying test metrics.

### Task 6: Evidence and final verification

**Files:**
- Create: `docs/evidence/live-cache/expanded-next-layer-predictor.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Produces: concise measured/simulated report, hashes, limitations, and reproducible commands.

- [ ] Generate per-model, selected-test, layer/phase/conversation/domain, latency, artifact, and 32-slot shadow summaries.
- [ ] Append decisions, commands, RED/GREEN evidence, selection, test opening, failures, and hashes to `PROJECT_LOG.md`.
- [ ] Run the full test suite, artifact/ledger parsing, SHA-256 reconciliation, `git diff --check`, and worktree status.
- [ ] Commit the verified expanded-predictor milestone.
