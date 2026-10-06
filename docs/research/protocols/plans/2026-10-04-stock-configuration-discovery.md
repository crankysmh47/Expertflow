# Stock Configuration Discovery Implementation Plan

**Goal:** Validate a usable stock plan, then find the strongest eligible stock
configuration for the current model and provide a reusable MoE/hardware method.

**Architecture:** Extend the existing compiler, evidence store and paired runner.
Keep the legacy Phase3 protocol intact; new acceptance receipts independently
bind fresh paired evidence to stock plans. Add a host-aware bounded search layer
after the new product validation passes.

**Tech Stack:** Python3.11+, stdlib, pytest/uv, pinned pristine llama.cpp, Windows.

**Spec:** docs/research/protocols/specs/2026-10-04-stock-configuration-discovery.md

## Global constraints

- Work on ef-v2; preserve user R1 edits, historical verdicts and native builds.
- Exact model/workload, F16KV and unmodified arithmetic; no speedup from quality loss.
- Freeze protocol/source/runtime/host identities before native retention.
- No native retry/discard or alternate gate after seeing results.
- Only declared measured search-space coverage can support a strongest-stock claim.

## Tasks

### Task1: historical speed and host audit

- [x] Create docs/evidence/stock-discovery-20261004/history-audit.json and report.md.
  Record source hashes, mode/interface/concurrency, stock22.9667, confirmation
 24.411/replay25.383, static28.13qualitySTOP and aggregate35.6699 distinction.
- [x] Capture CPU topology/identity, RAM, OS and power policy without changing them.
- [x] Establish the temporary study directory stock-configuration-discovery/progress.md as ledger
  for this complete user objective; retain native process handles there.

### Task2: evidence-backed paired stock-product acceptance

Files: src/expertflow/compiler/stock_validation.py, refinement.py;
tests/test_compiler_stock_validation.py.

- [x] Write failing tests for twenty verified fresh product-stage rows, correct
  schedule/candidate binding, forged statistics, old A/A rejection, partial
  failure/no publication, success publication and receipt revalidation.
- [x] Add product validation mode to existing execute_pairs without changing
  default A/A behavior. Freeze new protocol and complete source hashes; keep
  source evidence read-only. Record distinct product-stage IDs.
- [x] Add receipt publication/verification bound to plan/experiment/host and exact
  EvidenceStore measurements. Recompute statistics; require fresh stages and
  exact plan launch identities/settings; fail closed on altered/missing evidence.
- [x] Run targeted RED/GREEN and full uv dev/quality/predictor pytest suite.

### Task3: CLI and host-aware accepted execution

Files: commands.py, preflight.py, scripts/benchmark_compiler_refinement.py;
tests/test_compiler_cli.py and stock_validation tests.

- [x] Add stock-product experiment and source-plan/database arguments to existing
  benchmark driver; add optional acceptance receipt to validate/sealed run.
- [x] Capture and revalidate CPU/OS/RAM/power identities; preserve legacy paths.
- [x] Accepted execution checks fresh tokens/memory/cleanup and returns measured
  status without using the old single-sample absolute2% publication gate.
- [x] Verify CLI fail-closed behavior and full suite; commit before native runs.

### Task4: live stock-product validation

- [x] One fresh output/DB; exact pinned Gemma workload, source pending plan and
  original reviewed DB; freeze new stage/protocol/host pins.
- [x] Execute exactly20cold processes sequentially; inspect owned process state
  on timeout, never relaunch a live process. All failures are retained.
- [x] Independently reconstruct artifacts/pairs/stats; publish fresh plan and
  receipt only on PASS-STOCK-FALLBACK. Original result remains unchanged.
- [x] One final fresh review/fix pass and full checks; save report/artifact hashes.

### Task5: bounded stock discovery

- [x] Implement generic semantic workload/search environment/search space records
  and deterministic candidate generation with explicit numerical eligibility.
- [x] Freeze exact first search manifest from current CPU topology/capabilities,
  declared coverage, screen/confirmation budgets and stop rules before retention.
- [x] Screen eligible configurations, confirm finalist against incumbent with
  independent fixed balanced pairs, and accept only verified result.
- [x] Record older-speed comparison separately from contemporary paired gains.

### Task6: MoE/hardware reuse

- [x] Generalize model/runtime/host inputs through existing adapter registry;
  test multiple model inventories/topologies and identity invalidation.
- [x] Inventory available local model artifacts; add second real family/quant
  when supported weights exist, with independent normalization/quality evidence.
- [x] Document capability exclusions and live-tested coverage; audit full goal,
  not merely StageA, before claiming completion.

Terminal evidence: `docs/evidence/stock-discovery-20261004/objective-audit.md`
and `method-verification.json`. One actual Gemma family/two quantizations,
multiple synthetic family/topology contracts; universal live support is not
claimed. Q6 and Q4 retain12threads/graphs-on in their declared spaces. All100
registered native processes reverified; historical sources and verdicts intact.

## Review focus

- Forged receipts, old A/A rows, mismatched host/candidate/pair stages cannot pass.
- Partial native collection must not publish artifacts or consume retries.
- Screening winners require independent confirmation; old TPS are not controls.
- Unsupported model/runtime/quality settings fail explicitly, not generic success.
- CPU topology/power/runtime changes invalidate reuse even with unchanged GPU UUID.
