# Independent review and final fix pass

One fresh independent reviewer examined the phase's code, tests, registration,
documentation and preserved source evidence against base `20711c0` on `ef-v2`.
The initial review invocation failed before work at a service usage limit;
the same reviewer resumed and completed after the user's resume instruction.
No extra reviewer or re-review was used. The review was read-only and launched
no native model process.

The reviewer independently ran31 CPU boundary checks, with one expensive
reconstruction deselected, and rehashed all41 frozen files, six historical
pins and every archived source member. All matched. No critical finding or
deferred minor was reported. Two important public reporting findings were
accepted and fixed in one pass:

1. Search reports have neither `automatic_id` nor `default_id`; comparing the
   missing values emitted `selected_default: true`. Emit that utility-specific
   field only for two actual identities. Search now reports `retained_incumbent`
   and `recommended_id` separately. Both incumbent and alternate controls failed
   before the fix and passed afterwards.
2. Product scope lives in frozen plan identities; reference/search workloads
   live in different candidate fields. Reference/search validation replies have
   no report path. Extract the scope from each existing verified schema and the
   validated receipt, binding its bytes before/after driver execution. Product,
   reference/search validation, workload and changed-receipt regressions failed
   before the fix and passed afterwards.

All six new review regressions failed on the old implementation; the final
reporting/registration set passed33 tests after the fix. The source-bound
collectors/validators remain unchanged. Final full-suite and public read-only
validation records are separate from the first timing measurement.

## Review rulings

- Historical scientific validity and closed-study decisions remain under their
  original reports/audits, outside this implementation review. No old gate or
  claim was revised. Cost if wrong: implementation review does not independently
  recertify the scientific study.
- General validation speedup is not claimed: the timings are descriptive local
  observations. Cost if generalized: different hosts/evidence sets may not see
  the same reduction.
- Wider native utility and the future collector remain unverified. Registration
  is inspection only and reports execution not ready. Cost if mistaken for a
  result: four new scientific claims would lack native evidence.
- The reviewer ran focused CPU checks; the executor owns the full-suite evidence.
  Cost if misreported: focused checks cannot establish whole-repository success.
- The unrelated R1 note/result, old search preview and session transcript remain
  excluded. Cost if included: unrelated user changes would enter this checkpoint.

## Implementation rulings

- Scoped driver constructor substitution outside `compiler/` preserves every
  measured byte and source-map guard. Cost: the adapter depends on existing
  driver interfaces; raw read-only reconstruction and tests cover compatibility.
- Keep the active `ef-v2` checkout following the existing user preference.
  Cost: no new worktree isolation; explicit file staging and preserved unrelated
  diffs keep the checkpoint bounded. No merge, push or publication is performed.
