# ExpertFlow tasks

Current roadmap: [placement proof and stock fallback](superpowers/plans/2026-10-04-placement-proof-and-stock-fallback.md).
Results and scope: [STATUS.md](STATUS.md). Updated 2026-10-06.

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
- [x] Register and implement repeatability/acceptance under a separately reviewed
  protocol with a new justification and fixed budget before any further native run.
  [Follow-up protocol](superpowers/specs/2026-10-04-stock-repeatability.md) is
  authorized; all five review findings fixed, 797 tests/seven optional source skips
  and six pinned native source checks verified. Native collection completed
  all 148 calls from `86388e7`; [execution state](evidence/stock-repeatability-20261004/execution-state.md).
- [x] Qualify two independent paced acceptance blocks; stop on the first
  failed/inconclusive gate and independently reconstruct all retained evidence.
- [x] Collect both fresh consumers and the untouched transfer's full 107-call
  utility/product sequence; native gates passed at +9.38% held-out defaults gain.
- [x] Finish frozen outer reconstruction and final independent raw audit: PASS,
  148 native calls, 41 frozen files and six historical pins intact.
- [x] Finish fresh read-only CLI validation and verified evidence/documentation;
  exit0, no extra native calls, all source/history pins unchanged.
- [x] Consolidate the validated workflow into public CLI/decision reports.

Placement steps 2–3 are inactive after the feasibility rejection. The original
fallback remains **PRODUCT-VALIDATION-STOP**, with product CI90 [+0.13%, +2.34%]
outside the fixed ±2% margin. The separate follow-up has completed native gates
and independent raw reconstruction. Outer source/phase/receipt reconstruction
passed; the bounded method is qualified under fixed spacing/diagnostics.
See [follow-up results](evidence/stock-repeatability-20261004/report.md)
and [original terminal evidence](evidence/stock-utility-20261004/report.md).
Never reuse either study's budget for additional samples or candidates.

## Next product work

- [x] Reuse verified readers per database in public read-only validation;
  preserve measured source snapshots and all artifact/model checks.
- [x] Publish coverage, defaults scope, tuning costs and invalidation reasons
  with the CLI workflow. Keep neutral/default-optimal outcomes explicit.
- [x] Register broader utility coverage before new workload/model/host runs;
  Q4 and Granite compatibility does not establish defaults gain on those inputs.
- [x] Implement/review the separate wider collector and freeze its source,
  live identities and all four case roots before native collection.
  [Implementation plan](superpowers/plans/2026-10-05-wider-stock-collector.md):
  guarded per-case journal, complete/prefix reconstruction, public sequence
  commands and one independent implementation review.
- [x] Execute the [four registered cases](superpowers/specs/2026-10-05-stock-coverage.md)
  from `78ad5f0`; retain all 344 calls and four NO-UTILITY-GAIN outcomes.
  No conditional product/consumer ran; 84 unspent calls cannot fund retries.
- [x] Run fresh public read-only reconstruction: PASS, 93.40 seconds, zero new
  calls; all 63 source files and original 41 files/six history pins/148 records unchanged.
- [x] Finish independent raw audit: PASS, all 344 records/3,440 artifact hashes,
  no material discrepancies; publish the final wider evidence/docs.

Public CLI and reader-reuse verification are recorded in
[the implementation report](evidence/stock-cli-20261005/report.md). Wider native
results are [recorded separately](evidence/stock-coverage-20261005/report.md);
none met the practical utility gate. Serving and other hosts remain unverified.

## Next research gate; other expansion remains deferred

- [x] Audit wider offload/attention/batch operation paths and numerical
  eligibility: [scoped no-go](evidence/stock-control-scope-20261006/report.md),
  16 pinned upstream objects, zero native calls, 55 contract tests passed.
- [ ] Blocked native extension: establish complete exact arithmetic equivalence
  and a useful mechanism for a new control, or define a separate quality
  policy/provider, datasets and held-out gates. Register a new fixed budget
  before collection; do not retune or reuse the completed studies.
- [x] Test the explicit-enable attention mechanism prerequisite from pinned
  source: [no new supported fused-decode mechanism](evidence/stock-followthrough-20261006/report.md),
  492 existing launch records preserved and zero new model processes.
- [x] Continue the fail path into readable archived stock outcomes and delegated
  public verification; 55 focused helper/CLI/reader tests passed after one review,
  its fix pass and a live argument-binding correction. No compiler/provider changes.
- [x] Prepare executable status/verification commands and a five-task
  [independent-user handoff](stock-product-handoff.md).
- [x] Qualify both studies through fresh public read-only validation:
  [PASS-LOCAL-STOCK-QUALIFICATION](evidence/stock-followthrough-20261006/verification.json),
  331.65 seconds, zero extra native calls; failed initial attempt preserved.
- [ ] Obtain actual independent-user usability results; continue onboarding
  on pass, or fix the observed workflow problem on fail. Agent checks do not
  complete this product gate. The user attempted verification, then requested
  agent help with the questions; [state](evidence/stock-followthrough-20261006/usability-state.json)
  records no five-task unaided success or human timings.
- [x] Complete the requested Luna walkthrough and full agent-run reconstruction
  after authorized GPU-app closure: [PASS in 313.51 seconds](evidence/stock-agent-walkthrough-20261006/report.md),
  zero extra model calls, earlier blocked attempts retained. Add GPU-stop guidance
  without changing native guards or scientific thresholds.
- [ ] KV compression/TurboQuant and shared memory-budget optimization.
- [ ] MTP/speculation, two-table dynamic residency and joint search.
- [ ] Broad family/hardware coverage, serving integration and presentation polish.

After every decision, append the result to `PROJECT_LOG.md` and update this list,
`STATUS.md` and the relevant claims. Checkboxes indicate completed work only;
writing a protocol or passing CPU tests does not complete a native proof gate.
