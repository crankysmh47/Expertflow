# Next-Layer Shadow Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate bounded one-layer-ahead expert predictors on the frozen canonical 7/4/3 pilot without runtime integration.

**Architecture:** A strict dataset builder converts canonical router events into adjacent-layer samples. Deterministic baseline and learned predictors share one ranking interface, metrics, and a conversation-reset 32-slot shadow simulator. Validation freezes the family/features/seed/widths before one sealed test evaluation.

**Tech Stack:** Python 3.11+, standard library, NumPy, PyTorch CPU, pytest, canonical trace schema 1.0.0.

## Global Constraints

- Use only `trace_v2_canonical_segmented_pilot`; preserve its exact seven/four/three split.
- No llama.cpp, runtime, GPU, transfer, residency, or cache-decision changes.
- B0, B1, B2, B3, then at most one fixed B4; no search.
- Test metrics are read only after `selection-lock.json` is written.
- All cache results are simulated shadow evidence; no speedup or overlap claim.

---

### Task 1: Strict adjacent-layer dataset

**Files:**
- Create: `src/expertflow/predictor/dataset.py`
- Create: `tests/test_predictor_dataset.py`

**Interfaces:**
- Produces: `PredictionSample`, `PilotDataset`, and `load_pilot_dataset(manifest: Path) -> PilotDataset`.

- [x] Write failing tests with shuffled synthetic events that require same-conversation/token joins, true next-layer targets, immutable split membership, causal previous-token features, and explicit duplicate/missing/ambiguous failures.
- [x] Run `uv run pytest tests/test_predictor_dataset.py -q`; expect import failure.
- [x] Implement immutable dataclasses, strict grouping by `(conversation_id, forward_id, token_index, token_id)`, sorted adjacent-layer joins, 128-vector encoding, and split-disjoint assertions.
- [x] Rerun the focused test and full suite; expect green.

### Task 2: B0-B2 rankings and metrics

**Files:**
- Create: `src/expertflow/predictor/models.py`
- Create: `src/expertflow/predictor/metrics.py`
- Create: `tests/test_predictor_models.py`
- Create: `tests/test_predictor_metrics.py`

**Interfaces:**
- Produces: `CopyPredictor`, `FrequencyPredictor.fit(train)`, `TransitionPredictor.fit(train)`, `rank(sample) -> tuple[int, ...]`, and `evaluate_predictions(...)`.

- [x] Write failing tests proving B1/B2 learn from supplied training samples only and deterministic ties use ascending expert ID.
- [x] Write failing metric tests for recall@8/12/16, overlap, exact set, per-layer, phase, and conversation breakdowns.
- [x] Run focused tests and confirm missing imports/functions are the RED reason.
- [x] Implement minimal deterministic counters/rankings and metric aggregation; rerun focused and full tests green.

### Task 3: Fixed B3 and B4 CPU models

**Files:**
- Create: `src/expertflow/predictor/learned.py`
- Create: `tests/test_predictor_learned.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: fixed feature encoding, `LinearPredictor.fit`, `SharedMlpPredictor.fit`, `rank`, `parameter_count`, `save`, and measured batch-one latency.

- [x] Add the `predictor` optional dependency group for NumPy and PyTorch with bounded versions.
- [x] Write failing deterministic toy-learning tests for one `torch.nn.Linear` B3 and one fixed one-hidden-layer B4 using seed `20260716` and CPU only.
- [x] Run focused tests and verify RED due to missing implementation.
- [x] Implement weighted BCE, one fixed optimizer/epoch schedule, target-layer one-hot, phase bit, source vector, and causal previous-target vector; no tunable search loop.
- [x] Rerun learned and full tests green.

### Task 4: Shadow 32-slot simulator

**Files:**
- Create: `src/expertflow/predictor/shadow.py`
- Create: `tests/test_predictor_shadow.py`

**Interfaces:**
- Produces: `simulate_shadow(samples, rankings, width, capacity=32, expert_bytes=3345412)`.

- [x] Write failing causal examples distinguishing reactive hits, useful predictions, waste, uncovered misses, extra evictions, eviction regret, and bytes.
- [x] Run the focused test and confirm RED.
- [x] Implement paired conversation-reset LRU states with identical demand order and speculative insertion only into the target layer.
- [x] Rerun focused and full tests green.

### Task 5: Validation selection and sealed test

**Files:**
- Create: `scripts/run_next_layer_predictor.py`
- Create: `tests/test_predictor_pipeline.py`

**Interfaces:**
- Produces external `validation-metrics.json`, `selection-lock.json`, `test-metrics.json`, model files, and `ledger.jsonl`.

- [x] Write a failing integration test proving `test` refuses to run without a valid selection lock and `fit` never emits test metrics.
- [x] Implement `fit` and `test` subcommands, artifact hashes, exact split IDs, environment/seed/features, latency, parameter size, and append-only commands.
- [x] Run `fit` once on train/validation, select by validation success rule with simplicity tie-break, and write the immutable lock.
- [x] Run `test` once for the frozen family and widths 8/12/16; do not revise configuration afterward.

### Task 6: Evidence and final verification

**Files:**
- Create: `docs/evidence/live-cache/next-layer-shadow-predictor-pilot.md`
- Modify: `PROJECT_LOG.md`

- [x] Write one concise report containing separate validation/test metrics, per-layer/phase/conversation tables, shadow accounting, latency, parameters, artifact hashes, and small-pilot limitations.
- [x] Run all tests, parse every JSON/JSONL artifact, verify manifest hashes and split disjointness, run `git diff --check`, and confirm protected/runtime source worktrees remain untouched.
- [x] Commit the verified bounded pilot milestone without runtime integration.
