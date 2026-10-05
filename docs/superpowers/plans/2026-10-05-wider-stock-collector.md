# Wider stock collector implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one independent review before native collection.

**Goal:** Collect and reconstruct the four registered wider stock utility cases without modifying the closed Q6 collectors.

**Architecture:** A new project driver freezes the complete registered sequence and loads all live inputs before collection. A separate wider module uses the original native runner, eligibility providers, selection and statistics primitives, with its own registered scope, paced attempt journal and prefix reconstruction. Existing stock product acceptance remains authoritative.

**Tech stack:** Python standard library, SQLite evidence, pytest, existing Windows native runner.

**Spec:** [Registered protocol](../specs/2026-10-05-stock-coverage.md).

## Global constraints

- Stay on ef-v2; preserve unrelated files and all original frozen source/history files.
- Four cases in registered order; six candidates; 18 automatic and 18 manual evaluations, 86 utility calls and conditional 21 product/consumer calls per case; maximum 428 total.
- Wait at least 30 seconds before every call; four hours per case including input load and reconstruction, 16 hours overall; health/completion limits 180/300 seconds.
- No retry, resume, discards, replacement, pooling, new candidate or alternate finalist.
- Statistical failure stops one case; native correctness, identity, source, environment, memory, cleanup or resource failure stops the sequence.
- The registration and original protocol bytes remain immutable. CPU fixtures do not establish scientific coverage.
- User authorization covers implementation, review, input freeze and registered collection. Do not repeat approval questions.

## Review focus

1. A complete negative gate and an incomplete valid prefix must never be promoted to utility/product success.
2. All attempts, including failed starts and product/consumer calls, must bind unique owners, exact roots/databases and the outer freeze.
3. Source, host or input drift during the fixed wait must prevent the next launch.
4. Case and sequence wall caps must include loading/reconstruction and stop further launches; no hidden warmup calls.
5. Public read-only reconstruction must recompute selection/statistics and expose measured scope, actual costs and invalidation rather than trusting report fields.

## Task 1: Wider utility and guarded journal

Files: create `src/expertflow/stock/wider.py`, `src/expertflow/stock/wider_audit.py`, `tests/test_stock_wider.py`.

Interfaces: `execute_case(inputs, case, sequence, runner_factory, capture)` returns a persisted terminal report; `validate_case(report, inputs, sequence, capture)` reconstructs every retained native record and complete statistical gates. `PacedRunner` journals a call before waiting and binds it to the outer manifest.

- [ ] Write positive/neutral, incomplete prefix, changed statistics/costs/model/source, pacing and budget tests; observe expected failures.
- [ ] Implement separate registered scope and six-candidate collection using original pure utility primitives; never call or patch the Q6 scope guard.
- [ ] Reconstruct complete gates and incomplete prefixes, product/source/consumer binding and all attempts; reject missing/unaccounted starts.
- [ ] Run `uv run pytest -q tests/test_stock_wider.py`; expected all pass. Record RED/GREEN logs.

```python
assert neutral['status'] == 'NO-UTILITY-GAIN'
assert len(neutral['attempts']) == 86
assert 'consumer' not in neutral
assert incomplete['utility_gain_established'] is False
```

## Task 2: Sequence driver and public CLI

Files: create `scripts/benchmark_compiler_stock_coverage.py`, `tests/test_stock_wider_sequence.py`; modify `src/expertflow/stock/cli.py`; update README/STATUS/TODO/BENCHMARKING.

Interfaces: `run_sequence(registration_path)` freezes all four live inputs, roots, proofs, source commit/map and schedules before the first call; `validate_sequence(report_path)` independently reconstructs the sequence without native launches. Public commands are `expertflow stock coverage run` and `expertflow stock coverage validate`; `inspect` remains registration-only.

- [ ] Test full sequence, statistical continuation, environment/resource stop, fresh-root enforcement and validation-only native-call count; observe failures.
- [ ] Implement exact registered roots and sequence manifest, source-map guards, pooled readers, explicit per-case/phase/wait/load/reconstruction costs and existing sampler lifecycle.
- [ ] Run `uv run pytest -q tests/test_stock_wider.py tests/test_stock_wider_sequence.py tests/test_stock_cli.py tests/test_stock_coverage.py`; expected all pass.
- [ ] Verify installed wheel help and public coverage routing outside the repository; update docs with exact commands and pending collection status.
- [ ] Commit only task-owned files, then obtain one independent implementation review. Fix important findings with failing regression controls followed by passing focused/full checks.

```powershell
uv run expertflow stock coverage inspect
uv run expertflow stock coverage run
uv run expertflow stock coverage validate
```

## Task 3: Reviewed freeze, bounded execution and durable results

Files: create `docs/evidence/stock-coverage-20261005/` records; update README/STATUS/TODO/BENCHMARKING/PROJECT_LOG with verified outcomes.

- [ ] Verify reviewed source/map, original historical integrity, live weights/binaries/host, trusted eligibility/defaults and absence of foreign model processes. Freeze all roots/databases and both schedules before any call.
- [ ] Execute the four cases in registered order within fixed budgets, retaining every attempt and stopping as prescribed. Native logs live at registered absolute roots.
- [ ] Independently audit raw retained artifacts and run public read-only validation; verify unchanged historical study and native-start counts before/after validation.
- [ ] Record four separate measured/partial/unrun verdicts, selection, defaults/manual intervals and all costs; never infer a family-wide gain.
- [ ] Commit durable evidence/docs on ef-v2. Preserve ignored scratch if automatic cleanup review blocks deletion; do not evade that rejection.

```python
assert sequence['attempts'] <= 428
assert validation['additional_native_calls'] == 0
assert historical_integrity['unchanged'] is True
```
