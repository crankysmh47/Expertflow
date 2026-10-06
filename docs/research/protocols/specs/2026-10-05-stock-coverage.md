# Registered wider stock utility coverage

Status: **REGISTERED-NOT-RUN**. This registration defines the next four comparisons.
It does not extend the completed Q6 study, establish another gain, or authorize
reusing its budget. Machine-readable identities are in
[stock-coverage-20261005.json](../../../configs/compiler/stock-coverage-20261005.json).

## Question and cases

Does the same bounded thread/graph tuning method provide useful defaults gains
on Gemma Q4 and Granite Q6, while reproducing independently tuned stock? Test
each artifact with the existing prose prompt and existing code prompt. These
prompts have already been used in other studies; these cases are wider coverage,
not untouched-workload evidence. No different host is included.

| Order | Case | Model and fixed placement | Workload | Maximum calls |
| --- | --- | --- | --- | --- |
| 1 | gemma4-q4-prose | Q4_0, ngl99, CPU-MoE true | baseline prose, 512-token decode | 107 |
| 2 | gemma4-q4-code | same exact Q4 bytes and placement | code prompt, 512-token decode | 107 |
| 3 | granite-q6-prose | Q6_K, ngl99, CPU-MoE false, GPU resident | baseline prose, 512-token decode | 107 |
| 4 | granite-q6-code | same exact Granite bytes and placement | code prompt, 512-token decode | 107 |

The registration pins descriptors, complete inventories, prompt/workload files,
hardware/runtime input files, normalized model/workload identities, runtime
bindings, host environment, numerical provider and upstream defaults proof.
Model files remain at their original paths; registration inspects metadata,
not a fresh full-weight digest. All actual weight/binary/host/source checks
must pass again before the first native call.

## Method and budgets

Use six controls: threads 8, 12 and 16 crossed with CUDA graphs on/off. All
other controls remain fixed: pristine runtime, F16 KV, ctx4096, batch2048,
ub512, greedy seed42, ignore EOS, no prompt cache, concurrency1. Resolve only
native thread/graph defaults to 8/on from the pinned single-socket 8-core,
16-logical-processor host and upstream source. This is not a comparison with
all out-of-box flags; Granite's fixed full-GPU placement differs from Gemma's.

Per case: reference10 + automatic18 + independent manual18 + defaults pairs20
+ manual pairs20 = **86 utility calls**. If and only if utility passes, add
fresh product20 and one fresh accepted consumer: **107 maximum**. Freeze the
three seeded screening blocks separately for automatic and manual runs; use
the existing seed20261004 schedule and paired seed20261003, 10,000 bootstrap
resamples, nearest-rank intervals and geometric paired changes. Automatic
selection ranks three-block geometric ratios to defaults; manual selection
ranks the arithmetic mean of its own three evaluations per control. Both
tie-break toward defaults and then candidate ID. Never reuse owners/samples
between arms, grids, cases, confirmation, product or consumers.

Maximum across four cases: **428 calls**. Minimum fixed prelaunch wait30s
before every call, including reference and consumers; minimum scheduled wait
at the maximum budget is12,840s. Record actual waits separately. Use mandatory
process-owned/device-free memory sampling at0.2s with optional diagnostics.
No warmup native call, discard, replacement, retry, third block, additional
candidate, resume or alternate finalist. Retain all attempted calls, including
failed starts. Each case has a fresh output root and database; product has a
separate database. Collection wall cap is4hours per case/16hours overall,
including input loading, fixed waits and final reconstruction. Stop starting
calls at the wall cap; never present a partial case as a statistical result.
Per-call health180s/completion300s limits and owned cleanup remain mandatory.

## Gates and honest terminal outcomes

- Stable reference and each paired arm: CV at most10%; exact own-reference
  prompt/output tokens, identities, numerical eligibility, memory/reserve and
  owned-process cleanup must pass. No cross-model token comparison.
- Utility: paired geometric defaults gain at least5%, CI95 lower strictly
  above0; manual equivalence CI90 strictly inside[-2%,+2%]; automatic evaluation
  count no greater than manual, with18 evaluations each.
- Product: existing ten-pair equivalence CI90 strictly inside[-2%,+2%],
  one-sided95 lower strictly above-2%, arm CV at most10%, then fresh consumer.
- Default-optimal/neutral result: **NO-UTILITY-GAIN**, even when the selected
  configuration equals defaults and is reproducible. No accepted utility
  product or consumer follows a failed utility gate. State `selected_default`
  and preserve all86 utility records when collection completed.
- Variance, manual equivalence, numerical/token, memory, cleanup, source/host
  drift, budget or product failures retain their own stop reasons. Incomplete
  grids/pairs have no interval, winner promotion or utility verdict.

Stop the affected case at its first failed/inconclusive gate. Continue to the
next registered case only if failures are statistical and every raw record is
valid; identity/source/environment/memory/cleanup or resource-budget failure
stops the entire sequence until the prerequisite is resolved under a newly
documented execution decision. No pooling across cases or post-hoc selection
of favorable cases. Report four separate verdicts; each95% interval is a
per-case interval, not a simultaneous family-wide guarantee. A neutral result
is useful coverage evidence and is not grounds to retune the frozen search.

## Execution prerequisites

1. Implement a separate wider collector/validator outside the frozen original
   Q6 collectors. Include all new code, this specification and registration in
   its source freeze. Do not bypass the original Q6-only scope guard.
2. Verify CPU controls for positive gain, default-optimal ties, partial grids,
   altered statistics/costs, changed models/inputs, and process-budget stops.
   Fixtures are not measurements; no positive synthetic case can count as
   wider scientific coverage.
3. Obtain one independent implementation review and fix material findings.
   Freeze reviewed source commit, immutable source map, exact candidate IDs,
   both schedules, all case roots/databases, current weights/binaries/host and
   trusted eligibility/defaults proofs before collection. If a registered
   input or method changes, publish a new registration rather than editing
   this one after collection begins.
4. Audit all raw retained artifacts independently and run fresh public
   read-only validation. Report defaults/manual comparisons, selection,
   tuning18/18 cost, actual phase times, input/load/reconstruction/CLI times,
   waits, peak owned memory and minimum device-free reserve separately.

The original utility106/107 and repeatability148/148 studies stay closed.
The public `coverage inspect` command verifies this registration's small
input/protocol pins and reports `execution_ready: false`. It launches no model.
Wider collection and utility gains remain pending the prerequisites above.
