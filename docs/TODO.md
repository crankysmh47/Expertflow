# ExpertFlow tasks

Current roadmap: [placement proof and stock fallback](superpowers/plans/2026-10-04-placement-proof-and-stock-fallback.md).
Results and scope: [STATUS.md](STATUS.md). Updated 2026-10-04.

## Delivered

- [x] Compiler spine and owned/evidence-bound runtime validation.
- [x] Fresh accepted stock plans and bounded searches for Gemma Q6 and Q4.
- [x] Real Granite MoE adapter, numerical scope, reference, acceptance and search.
- [x] Close thread/prefetch/PDL/cache experiments with their original verdicts.
- [x] Align current documentation with accepted results and proof-first priorities.

## Closed placement feasibility; inactive experiment branch

- [x] Audit numerical-path differences and reconstruct the failed held-out quality gate.
- [x] Document scoped placement no-go (plan step 1): no new exact mechanism under pinned kernels.
- [ ] Freeze matched controls, numerical/quality policy, datasets, all native
  process budgets and acceptance gates before a new experiment (step 2).
- [ ] Run one bounded experiment and independent raw-evidence audit; retain failures.
- [ ] If it passes, demonstrate automatic compiler selection, sealed execution,
  and a separately budgeted held-out transfer test (step 3).

## Executed stock-autotuning fallback

- [x] Activate after scoped placement no-go; current claim remains validated selection/reproduction.
- [x] Freeze defaults/manual-tuning comparisons, search costs, eligible controls
  and unseen test inputs (step 4).
- [x] Implement and review the collector; reproduce/fix all three review
  findings; verify 767 tests and six applicable native source checks.
- [x] Execute and independently audit the frozen main comparison: +12.70%
  over resolved defaults, manual equivalence, equal 18-evaluation grids.
- [x] Attempt fresh paired product acceptance and retain its inconclusive
  result; close collection at 106/107 processes without retries/discards.
- [x] Choose the narrower validated selection/reproduction product scope
  pending end-to-end acceptance; report the narrow default-tuning gain.
- [ ] Resolve repeatability/acceptance under a separately reviewed protocol
  with a new justification and a fixed budget before any further native run.
- [ ] After full acceptance, validate a fresh consumer and the untouched
  transfer workload, then consolidate public CLI/decision reports.

Placement steps 2–3 are inactive after the feasibility rejection. The fallback
study reached **PRODUCT-VALIDATION-STOP**: product CI90 [+0.13%, +2.34%] was not
within the fixed ±2% equivalence margin. Consumer, transfer and CLI expansion
are blocked by this gate. Both prompts were frozen, but the transfer prompt
remains unmeasured. See [terminal evidence](evidence/stock-utility-20261004/report.md).
Do not reuse the closed study's unused budget to retry acceptance.

## Deferred until a proof path passes

- [ ] Wider offload/attention/batch controls with their numerical scope proofs.
- [ ] KV compression/TurboQuant and shared memory-budget optimization.
- [ ] MTP/speculation, two-table dynamic residency and joint search.
- [ ] Broad family/hardware coverage, serving integration and presentation polish.

After every decision, append the result to `PROJECT_LOG.md` and update this list,
`STATUS.md` and the relevant claims. Checkboxes indicate completed work only;
writing a protocol or passing CPU tests does not complete a native proof gate.
