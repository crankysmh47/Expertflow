# Placement proof and stock-tuning fallback plan

> **For agentic workers:** Use superpowers:executing-plans task by task. This is a planning checkpoint; implementation and native measurements are separate work.

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

- [ ] On placement no-go, define the product as a hardware/workload-aware stock configuration compiler with the claim **best validated within the searched space**.
- [ ] Admit only reviewed controls. Threads/graphs have current live coverage; offload, batching, attention and KV changes require their own numerical or quality eligibility before search.
- [ ] Freeze a benchmark against out-of-box defaults and a documented manual-tuning baseline, using matched inputs and declared equal search budgets. Proposed utility gate: **at least 5% paired TPS gain over defaults with 95% lower bound above 0**, and **equivalence within 2% of the manual result at no greater measured tuning cost**.
- [ ] Test the method on an unseen workload or host; keep untouched test inputs separate from search. Report neutral/default-optimal cases and costs instead of promising an improvement on every input.
- [ ] Only after utility passes, consolidate scripts into the public CLI, provide decision explanations and invalidation, and verify fresh consumer execution. If utility fails, scope the product as configuration validation/reproduction rather than acceleration.

**Exit:** measured stock-tuning utility with honest coverage and cost, or an explicit narrower product decision.

## Tracking and verification

- [x] Align current README, architecture, benchmarking, historical guide notices, status and task list with accepted evidence and this decision order.
- [ ] After each stage, append its result to `PROJECT_LOG.md`, update `docs/STATUS.md` and `docs/TODO.md`, and link raw/audit artifacts. Change public claims only after the corresponding gate passes.
- [ ] Defer KV compression/TurboQuant, MTP/speculation, dynamic residency, universal adapters and product polish until a proof path passes. Existing accepted stock recommendations remain usable.

For future code changes, run focused checks, then the full gate once after final fixes:

```powershell
uv sync --frozen --extra dev --extra quality --extra predictor
uv run --no-sync pytest -q
uv run --no-sync python -m compileall -q src/expertflow
git diff --check
```

Run applicable native source checks against their exact external checkout. CPU tests and evidence replay do not establish a live speed or quality result.
