# ExpertFlow tasks

Current roadmap: [placement proof and stock fallback](superpowers/plans/2026-10-04-placement-proof-and-stock-fallback.md).
Results and scope: [STATUS.md](STATUS.md). Updated 2026-10-04.

## Delivered

- [x] Compiler spine and owned/evidence-bound runtime validation.
- [x] Fresh accepted stock plans and bounded searches for Gemma Q6 and Q4.
- [x] Real Granite MoE adapter, numerical scope, reference, acceptance and search.
- [x] Close thread/prefetch/PDL/cache experiments with their original verdicts.
- [x] Align current documentation with accepted results and proof-first priorities.

## Next: placement proof

- [x] Audit numerical-path differences and reconstruct the failed held-out quality gate.
- [x] Document scoped placement no-go (plan step 1): no new exact mechanism under pinned kernels.
- [ ] Freeze matched controls, numerical/quality policy, datasets, all native
  process budgets and acceptance gates before a new experiment (step 2).
- [ ] Run one bounded experiment and independent raw-evidence audit; retain failures.
- [ ] If it passes, demonstrate automatic compiler selection, sealed execution,
  and a separately budgeted held-out transfer test (step 3).

## Conditional: stock-autotuning fallback

- [x] Activate after scoped placement no-go; current claim remains validated selection/reproduction.
- [ ] Freeze defaults/manual-tuning comparisons, search costs, eligible controls
  and unseen test inputs (step 4).
- [ ] Prove utility or choose the narrower validation/reproduction product scope.
- [ ] After utility passes, unify public CLI/decision reports and validate a
  fresh consumer run; historical experiment budgets remain closed.

Placement steps 2–3 are inactive after the feasibility rejection. The fallback
protocol freezes two workloads and up to 107 native processes each; the second
starts only after all first-workload gates pass. CPU fixtures do not prove utility.

## Deferred until a proof path passes

- [ ] Wider offload/attention/batch controls with their numerical scope proofs.
- [ ] KV compression/TurboQuant and shared memory-budget optimization.
- [ ] MTP/speculation, two-table dynamic residency and joint search.
- [ ] Broad family/hardware coverage, serving integration and presentation polish.

After every decision, append the result to `PROJECT_LOG.md` and update this list,
`STATUS.md` and the relevant claims. Checkboxes indicate completed work only;
writing a protocol or passing CPU tests does not complete a native proof gate.
