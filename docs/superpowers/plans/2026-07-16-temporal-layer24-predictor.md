# Temporal Layer-24 Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and seal a leakage-safe offline predictor for layer-24 experts at decode token `t+1` from the same layer's experts at token `t`.

**Architecture:** Add temporal-specific dataset, policy, metrics, and shadow modules rather than changing the accepted adjacent-layer implementation. A guarded CLI materializes train/validation during fit, writes an immutable selection lock, and opens the frozen test split exactly once.

**Tech Stack:** Python 3.12, standard library, pytest, canonical Observer v1 JSONL traces, JSON/JSONL evidence, SHA-256.

## Global Constraints

- Preserve P2 commits `0974a32` and `d8354b17`; do not merge or push.
- Use exactly the frozen 60/12/12 conversation identities and six-domain balance.
- Decode only, layer 24 only, consecutive tokens only.
- Fit on training conversations, select on validation, evaluate test once.
- Evaluate only T0.0 through T0.3, widths 8/12/16, and the four fixed combined-weight triples.
- Do not modify llama.cpp during T0.
- Keep `live_cache_enabled=false` and make no speedup claim.

---

### Task 1: Temporal dataset

**Files:**
- Create: `src/expertflow/predictor/temporal_dataset.py`
- Create: `tests/test_temporal_dataset.py`

**Interfaces:**
- Produces: `TemporalSample`, `TemporalDataset`, and `load_temporal_dataset(manifest_path, *, expected_split_counts, expected_domain_counts, require_unique_prompt_hashes, materialize_splits)`.

- [ ] **Step 1: Write failing causal-join tests**

Create fixtures with prefill plus three decode layer-24 records. Assert exactly
two samples, source/target token identity, router-order preservation, and sealed
split identity without reading its missing trace.

- [ ] **Step 2: Write failing rejection tests**

Reject duplicate layer-24 records, forward gaps, token-index gaps, turn or
request changes, non-monotonic hook order, wrong layer, non-decode pairing,
non-eight-wide sets, out-of-range IDs, duplicate conversations, split drift,
domain drift, and prompt-hash duplication.

- [ ] **Step 3: Run RED**

Run:
`uv run pytest -q tests/test_temporal_dataset.py`

Expected: collection fails because the temporal module does not exist.

- [ ] **Step 4: Implement the minimal strict loader**

Stream only requested split traces, validate the frozen manifest contract, keep
only decode layer-24 events, sort by hook order within each conversation, and
construct a sample only for exact `+1` forward/token progression. Fail on every
ambiguous sequence instead of skipping it.

- [ ] **Step 5: Run GREEN**

Run:
`uv run pytest -q tests/test_temporal_dataset.py`

Expected: all temporal dataset tests pass.

### Task 2: Temporal policies

**Files:**
- Create: `src/expertflow/predictor/temporal_models.py`
- Create: `tests/test_temporal_models.py`

**Interfaces:**
- Produces: `TemporalCopyPredictor`, `TemporalSessionFrequencyPredictor`,
  `TemporalTransitionPredictor`, `TemporalCombinedPredictor`, and
  `rank_conversation(samples, predictor)`.

- [ ] **Step 1: Write failing deterministic-policy tests**

Assert copy ordering, conversation-reset session counts, training-only
source-normalized transitions, deterministic zero-support completion, fixed
combined-weight validation, and full 128-ID permutation rankings.

- [ ] **Step 2: Run RED**

Run:
`uv run pytest -q tests/test_temporal_models.py`

Expected: collection fails because the policy module does not exist.

- [ ] **Step 3: Implement minimal policies**

Keep mutable session state outside serialized trained models. Update session
counts with the current source before ranking, reset on conversation change,
and expose transition scores so the combined predictor can reuse them without
duplicating tables.

- [ ] **Step 4: Run GREEN**

Run:
`uv run pytest -q tests/test_temporal_models.py`

Expected: all temporal policy tests pass.

### Task 3: Metrics and temporal shadow

**Files:**
- Create: `src/expertflow/predictor/temporal_metrics.py`
- Create: `src/expertflow/predictor/temporal_shadow.py`
- Create: `tests/test_temporal_metrics.py`
- Create: `tests/test_temporal_shadow.py`

**Interfaces:**
- Produces: `evaluate_temporal_predictions(samples, rankings, *, widths)` and
  `simulate_temporal_shadow(samples, rankings, *, width, capacity, expert_bytes)`.

- [ ] **Step 1: Write failing metric and cache-accounting tests**

Use hand-calculated sequences to verify recall/exact-set breakdowns, reactive
LRU misses, ready coverage, useful/wasted bytes, speculative evictions, and
eviction regret with conversation resets.

- [ ] **Step 2: Run RED**

Run:
`uv run pytest -q tests/test_temporal_metrics.py tests/test_temporal_shadow.py`

Expected: collection fails because the metric modules do not exist.

- [ ] **Step 3: Implement the calculators**

Require one 128-ID permutation per sample. In the shadow, apply predictions
after source token `t` and before target token `t+1`, compare against a separate
reactive 32-slot LRU, then demand the true target IDs.

- [ ] **Step 4: Run GREEN**

Run:
`uv run pytest -q tests/test_temporal_metrics.py tests/test_temporal_shadow.py`

Expected: all temporal metric tests pass.

### Task 4: Guarded validation selection

**Files:**
- Create: `scripts/run_temporal_layer24_predictor.py`
- Create: `tests/test_temporal_pipeline.py`

**Interfaces:**
- Produces commands:
  `fit --manifest PATH --output PATH` and
  `test --manifest PATH --output PATH`.

- [ ] **Step 1: Write failing pipeline tests**

Require refusal without a selection lock, refusal to refit an existing lock,
fixed policy/weight/width tie-breaks, lock-payload hash verification, selected
artifact hash verification, sealed identity verification, and refusal of a
second test opening without changing existing metrics.

- [ ] **Step 2: Run RED**

Run:
`uv run pytest -q tests/test_temporal_pipeline.py`

Expected: collection fails because the CLI does not exist.

- [ ] **Step 3: Implement fit**

Materialize train/validation only. Fit T0.2, construct the four T0.3 policies,
evaluate every fixed policy and width, measure batch-one latency, select with
the predeclared key, serialize only the selected trained state, then write
validation metrics and a hashed lock with `test_opened=false`.

- [ ] **Step 4: Implement test**

Verify all hashes and frozen IDs, materialize test only, evaluate the selected
policy once, write test metrics, mark the lock opened, and refresh the artifact
index. Append command start/failure/end records to `ledger.jsonl`.

- [ ] **Step 5: Run GREEN**

Run:
`uv run pytest -q tests/test_temporal_pipeline.py`

Expected: all temporal pipeline tests pass.

### Task 5: Freeze and run T0

**Files:**
- Modify: `PROJECT_LOG.md`
- Create after measurement: `docs/evidence/live-cache/temporal-layer24-predictor.md`
- Generate: `C:\models\expertflow\runs\temporal-layer24-predictor\*`

**Interfaces:**
- Consumes: the frozen canonical manifest and guarded CLI.
- Produces: validation metrics, selection lock, selected model, one sealed-test result, ledger, artifact index, and written recommendation.

- [ ] **Step 1: Run focused and full verification**

Run:
`uv run pytest -q tests/test_temporal_dataset.py tests/test_temporal_models.py tests/test_temporal_metrics.py tests/test_temporal_shadow.py tests/test_temporal_pipeline.py`

Then run:
`uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Fit validation and inspect the unopened lock**

Run:
`uv run python scripts/run_temporal_layer24_predictor.py fit --manifest C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json --output C:\models\expertflow\runs\temporal-layer24-predictor`

Verify the recorded test IDs match the frozen corpus and
`test_opened=false`.

- [ ] **Step 3: Open the temporal test split exactly once**

Run:
`uv run python scripts/run_temporal_layer24_predictor.py test --manifest C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json --output C:\models\expertflow\runs\temporal-layer24-predictor`

Then intentionally rerun and require a fail-closed “already evaluated” error
without changing the test-metrics hash.

- [ ] **Step 4: Document and log**

Report validation/test metrics by conversation and domain, latency, simulated
miss coverage, useful/wasted bytes, evictions, regret, limitations, and the
T1 go/no-go. Label cache results simulated and preserve the project goal:
creating enough lead time for a ready-useful exact asynchronous transfer.

- [ ] **Step 5: Verify branch state**

Run `git diff --check`, the complete test suite, judge replay, artifact hashes,
and process cleanup. Do not commit the temporal milestone without explicit
authorization.

