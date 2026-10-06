# Stock validation, public CLI and coverage registration

**Goal:** Reduce repeated model hashing, expose the existing stock workflows through `expertflow stock`, and register wider utility comparisons before collecting them.

**Architecture:** Keep the measured compiler modules and collectors byte-for-byte intact. An adapter outside `compiler/` loads the existing project drivers and scopes one verified EvidenceStore per database to a read-only validation invocation. Installed CLI help works everywhere; execution requires an explicit research checkout and the original identity-bound inputs.

**Tech stack:** Python standard library, existing argparse CLI, SQLite evidence, pytest.

**Spec:** This document implements the three next actions in the [approved roadmap](2026-10-04-placement-proof-and-stock-fallback.md) and the user's 2026-10-05 instruction to continue all three.

## Constraints

- Work on `ef-v2`; preserve unrelated files and historical studies.
- Do not change any of the repeatability study's 41 frozen files, gates or receipts.
- Reuse reader instances only; never memoize measurements, artifacts, runtime checks or statistical verdicts.
- Keep the existing model size/mtime/ctime/hash guard and comparison-cycle detection.
- No native collection in this phase. Wider registration is evidence of a planned test, not utility gain.
- Reference/search/product collection retains each existing driver's explicit action, fresh outputs, budget and eligibility guards. Closed utility/repeatability studies expose validation only.

## Review focus

1. Model replacement after a cached verification must fail on the next check.
2. Rechecking an artifact must still detect tampering; no validation result cache.
3. Driver aliases, import paths and working directory must be restored after exceptions.
4. Installed CLI must explain missing checkout/input prerequisites without repository-only imports at startup.
5. Neutral, incomplete and invalid results must not acquire a performance claim through reporting.

## Task 1: Verified reader reuse

Files: `src/expertflow/stock/readers.py`, `tests/test_stock_readers.py`.

Interface: `EvidenceReaders(path)` returns the same EvidenceStore for each resolved database during one invocation; `reuse_readers(*modules)` restores each module's original constructor on exit.

- [x] Write tests for one model digest per database, stat invalidation, repeated artifact verification, isolated invocations, and restoration after exceptions; observe failures.
- [x] Implement the scoped reader pool, leaving `compiler/evidence.py` unchanged.
- [x] Reconstruct the existing completed fixture with pooled readers and retain the original source map.

## Task 2: Public CLI

Files: `src/expertflow/stock/cli.py`, `src/expertflow/cli/main.py`, `tests/test_stock_cli.py`.

Interface: `expertflow stock [--project PATH] <reference|product|search|utility|repeatability> <action> [original driver flags]`; `coverage inspect --registration PATH` reads the wider registration without launching anything.

- [x] Test routing, missing checkout, forbidden override flags, closed-study run rejection, honest result reports, and wheel installation help; observe failures.
- [x] Delegate to exact project drivers with no shell command composition. Translate product collection to `--experiment stock-product`; retain existing public `validate --acceptance` and `run --plan --acceptance` consumers.
- [x] Add decision scope, cost information and invalidation reason to the JSON result. Mark unverified input summaries explicitly.
- [x] Time read-only repeatability reconstruction and verify the final reviewed adapter; check all 148 records, frozen source/history hashes and zero additional native starts.

## Task 3: Wider registration and documentation

Files: `configs/compiler/stock-coverage-20261005.json`, new Q4/Granite workload configurations, `docs/research/protocols/specs/2026-10-05-stock-coverage.md`, README, STATUS, TODO, BENCHMARKING, stock method and project log.

- [x] Register Q4 and Granite, each with existing prose and code prompts: four separately bounded 107-call comparisons, 428 calls maximum, no pooling or retries.
- [x] Pin model, inventory, workload/prompt, runtime, host, defaults proof, candidate controls, costs, gates and stop rules; explicitly require a reviewed wider collector and frozen implementation before execution.
- [x] Add contract controls for positive, neutral/default-optimal, partial and tampered outcomes; distinguish CPU fixtures from scientific results.
- [x] Run focused compiler/CLI tests and packaging checks; request one final independent review, fix material findings with regression tests, then update durable execution evidence.

Execution note: authorization already covers these three stages. Additional design permission is not required. Existing measured source and original artifact paths remain authoritative; the CLI adapter does not relax their checks.
