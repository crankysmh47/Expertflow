# Placement proof and stock-tuning fallback plan

> **For agentic workers:** Use the implementation checklist task by task. This is a planning checkpoint; implementation and native measurements are separate work.

**Goal:** Establish useful compiler-selected MoE acceleration before product expansion; pivot to validated stock autotuning if placement cannot qualify.

**Architecture:** Reuse model adapters, profiling, static candidate selection, the owned runner and EvidenceStore. First prove a placement mechanism, then automatic selection and transfer. The fallback uses pristine llama.cpp and separately eligible stock controls.

**Tech stack:** Python, pytest, SQLite, pinned llama.cpp/CUDA, Windows/NVIDIA.

**Design basis:** [Compiler design](../specs/2026-08-28-expertflow-inference-compiler-design.md) and the agreed proof-first direction. This plan changes next-work priority; it does not change historical scientific verdicts.

## Constraints and proposed new gates

- Stay on `ef-v2`; isolate native source/build changes and preserve unrelated files.
- Historical static `QUALITY STOP`, Phase 3 `VALIDATION-STOP`, and cache/thread/prefetch/PDL results remain closed. No repeat of a failed hypothesis without a new, evidenced mechanism.
- Match model bytes, host, runtime interface, workload, KV, concurrency and quality policy. Charge both compiler and comparison search costs to their budgets.
- Selection data and held-out confirmation/quality/transfer data are separate. A hand-selected winner proves a mechanism, not automatic compiler selection.
- Exact execution requires numerical-path eligibility and native tokens. CPU-to-CUDA numerical changes require a separately labeled quality-bounded protocol; finite token parity cannot establish exactness.
- Proposed placement acceptance: paired geometric decode-TPS gain **at least 10%**, bootstrap **95% lower bound above 0**, each arm **CV at most 10%**, owned memory/reserve and cleanup PASS, and sealed product acceptance. Use the existing +1% PPL upper-confidence limit for a quality-bounded profile; declare a task-quality gate before collection.
- These thresholds govern new protocols only. Freeze source, dataset identities, complete process budget and rejection rules before any new native measurements; no retries, discarded ordinary samples, extra finalists or threshold changes after seeing results.

## 1. Qualify one placement hypothesis

**Read:** `docs/evidence/q6-placement-final/{report.md,quality-results.json}`, `docs/evidence/compiler-phase3/README.md`, and `docs/evidence/compiler-phase-profile-20261004/report.md`.

**Inspect:** `src/expertflow/compiler/{static_placement.py,pipeline.py,cost_model.py}`, `release/expertflow-build-week/patches/llama.cpp/`, and the pinned external native source.

- [x] Audit why placement changes numerical execution and why the original quality confidence gate failed. The favorable PPL point estimate does not establish a defect or justify simply taking more samples.
- [ ] Select one concrete mechanism justified by that audit and one memory-constrained MoE workload. Start with the verified Gemma Q6 artifact if the mechanism applies; fully resident Granite establishes stock compatibility only.
- [x] Write `docs/evidence/placement-proof-20261004/feasibility.md`: numerical policy, achievable memory relief, source boundary, required instrumentation, and a pass/reject decision. No new eligible exact mechanism exists in scope; proceed to step 4.

**Exit:** a specific testable mechanism and numerical contract, or a documented placement no-go. This step performs read-only analysis, with no native inference.

Execution result: scoped no-go, with source hashes and exact historical NLL gate
reconstruction. Steps 2–3 are inactive; step 4 is the active proof path.

## 2. Freeze and test the mechanism

**Deliverables:** `docs/evidence/placement-proof-20261004/{protocol.md,report.md}` and immutable raw evidence under a fresh `C:/models/expertflow/runs/` directory.

- [ ] Freeze controls, actual source/model/host hashes, selection and held-out inputs, metrics, all gates, and the full native-process budget. Limit selection to **six configurations and three complete screening blocks**, then **one finalist and ten fresh balanced confirmation pairs**. Itemize reference, numerical, quality and sealed-replay processes in the same total budget.
- [ ] Reuse the accepted strongest stock configuration as a starting control; validate contemporary matched controls and document remaining stock-space limits. Do not compare directly with historical 22.967/24.411/28.13 TPS summaries.
- [ ] If source changes are needed, add focused failing numerical/identity/boundary tests before implementing them in isolated source/builds. Run applicable native source checks and focused compiler tests before collection.
- [ ] Execute the frozen sequence once. Reconstruct statistics and native token, memory, ownership and cleanup checks from raw artifacts; stop on failed gates and retain every sample.

**Exit:** a quality-qualified mechanism meeting the proposed placement gate, or terminal no-go/inconclusive and step 4. A faster but quality-ineligible result fails.

## 3. Prove automatic selection and limited transfer

**Relevant code:** `src/expertflow/compiler/{static_placement.py,cost_model.py,pipeline.py,plan.py,runner.py}`; relevant existing tests are `tests/test_compiler_static_placement.py`, `tests/test_compiler_pipeline.py`, and `tests/test_compiler_plan.py`.

- [ ] Make the compiler derive candidates and select the winning plan from inventory, measured profile and memory constraints. Preserve all candidates, ranks, rejects and search cost; reject manual layer overrides from the proof path.
- [ ] Confirm the frozen compiler-selected plan on held-out runs, then validate a fresh sealed-plan consumer execution. Previously measured discovery samples cannot serve as independent confirmation.
- [ ] Freeze a separate bounded transfer test on one held-out memory-constrained model or workload. Apply the same selection method and quality rules without hand-tuning it for that test.
- [ ] Report tuning wall time and inference time separately, including amortization and startup overhead. Limit the acceleration claim to passing configurations and test scope.

**Exit:** accepted compiler-selected gain plus declared transfer coverage. Failure to transfer narrows the claim; it does not erase a valid narrow mechanism result. Freeze each new stage's total process budget before launching it.

## 4. Conditional stock-autotuning product

**Reuse:** `src/expertflow/compiler/{stock_reference.py,stock_validation.py,stock_eligibility.py,stock_search.py}`, `scripts/benchmark_compiler_stock_search.py`, and `docs/stock-configuration-method.md`.

- [x] On scoped placement no-go, activate stock utility evaluation. The current supported claim remains validated configuration selection/reproduction; acceleration claims depend on utility passing.
- [x] Admit only reviewed controls. Threads/graphs have current live coverage; offload, batching, attention and KV changes require their own numerical or quality eligibility before search.
- [x] Freeze a benchmark against resolved thread/graph deployment defaults and a documented independent manual grid, using matched inputs and equal native evaluation budgets. Utility gate: **at least 5% paired TPS gain over defaults with 95% lower bound above 0**, and **equivalence within 2% of the manual result at no greater measured tuning cost**. See the [frozen protocol](../specs/2026-10-04-stock-utility-proof.md) for the narrower default and cost scopes.
- [x] Implement/review the collector, fix all review findings, execute the main
  comparison once and independently audit all raw records. Utility comparison
  passed; fresh product equivalence was inconclusive. Retain the complete
  partial result and current validation/reproduction scope.
- [x] Test the unchanged method on the previously untouched code workload under
  the separate repeatability protocol; report its costs and declared scope.
- [x] Qualify both independent main blocks and fresh main/transfer consumers,
  with full held-out utility/product acceptance and independent raw audit.
- [x] Consolidate scripts into public CLI/decision explanations and invalidation.
  Preserve source snapshots and register new live validation budgets first.

**Exit:** measured stock-tuning utility with honest coverage and cost, or an explicit narrower product decision.

Execution exit: **PRODUCT-VALIDATION-STOP**, 106/107 native processes. Defaults
gain +12.70%, CI95 [+12.45%, +12.99%]; manual CI90 [-0.79%, +0.22%]; each search
used 18 evaluations. Product CI90 [+0.13%, +2.34%] exceeded the fixed +2% upper
equivalence limit. Correctness/identity/memory/cleanup and independent raw audit
passed. No consumer, transfer, retry or accepted new artifact. See
[terminal report](../../evidence/stock-utility-20261004/report.md).

The separately registered [repeatability/transfer protocol](../specs/2026-10-04-stock-repeatability.md)
subsequently passed 148 new calls and final reconstruction from 86388e7. Main
A/B CI90 [-0.553%,+0.540%]/ [-0.641%,+1.433%] passed independently, then the fresh
main consumer passed. Untouched-workload gain was 9.38%, CI95 [7.92%,10.61%],
manual CI90 [-0.389%,+0.645%],18 search evaluations each. Fresh transfer product
CI90 [-0.284%,+1.439%] and consumer passed; the independent raw audit agreed.
See [qualified result and costs](../../evidence/stock-repeatability-20261004/report.md).
Original studies, thresholds and quality/no-go verdicts remain closed.

The bounded stock-autotuning proof is complete under fixed spacing/diagnostics.
Public CLI consolidation and source-preserving reader reuse are implemented;
wider utility coverage completed the [four registered comparisons](../../evidence/stock-coverage-20261005/report.md)
from `78ad5f0`: 344 retained calls and four NO-UTILITY-GAIN outcomes, with
fresh public reconstruction and final independent raw audit passing.
Q4's gains were below the 5% gate; Granite retained the default. No new product
or consumer followed. The read-only offload/attention/batch numerical-scope
audit is [complete at a scoped no-go](../../evidence/stock-control-scope-20261006/report.md).
Further native work is blocked by missing numerical qualification and a useful
mechanism for another control, or a separate quality policy and acceptance
protocol. These results establish no global optimum, superiority to manual
tuning, new placement acceleration or serving throughput.

## Tracking and verification

- [x] Align current README, architecture, benchmarking, historical guide notices, status and task list with accepted evidence and this decision order.
- [x] Record both executed stages in `PROJECT_LOG.md`, `docs/STATUS.md` and
  `docs/TODO.md`, with raw/audit artifacts and qualified public claims.
- [x] Defer KV compression/TurboQuant, MTP/speculation, dynamic residency,
  universal adapters and product polish until complete proof. Existing accepted
  stock recommendations remain usable.

For future code changes, run focused checks, then the full gate once after final fixes:

```powershell
uv sync --frozen --extra dev --extra quality --extra predictor
uv run --no-sync pytest -q
uv run --no-sync python -m compileall -q src/expertflow
git diff --check
```

Run applicable native source checks against their exact external checkout. CPU tests and evidence replay do not establish a live speed or quality result.
