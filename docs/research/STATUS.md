# ExpertFlow current status

Updated 2026-10-06. Branch: `ef-v2`. Prior accepted evidence checkpoint:
`a793889`; measured Granite implementation: `7cbae5d`.

## Current decision

The user authorized implementation of the [local product roadmap](TODO.md) on
2026-10-06: portable setup, baseline profiles, bounded tuning, run/serve,
packaging and independent acceptance. Windows/NVIDIA is the first qualification
target; optional TUI/research expansion follows demonstrated needs. This direction
does not reopen closed studies or establish user readiness by itself. The current
supported result is validated stock selection/reproduction plus bounded Q6
autotuning utility over resolved defaults. An accepted new quality-preserving
speedup over already tuned stock has not been demonstrated.

Placement feasibility: **scoped no-go**, from a read-only numerical-path audit.
No new exact mechanism qualifies under the pinned CPU/CUDA kernels, and the
unchanged quality-bounded hypothesis is not reopened. See the
[audit and reconstruction](evidence/placement-proof-20261004/feasibility.md).

The stock utility study is complete at **PRODUCT-VALIDATION-STOP**, using
106 of 107 permitted native processes from measured source `0370579`.
The main utility gate passed at +12.70% over resolved thread/graph defaults,
CI95 [12.45%, 12.99%], with manual equivalence and equal 18-evaluation search
budgets. Fresh product equivalence was inconclusive: CI90 [+0.13%, +2.34%]
exceeded the fixed +2% upper limit. All native correctness, identity, memory
and cleanup checks passed; the independent raw audit reproduced the result.
See the [result](evidence/stock-utility-20261004/report.md) and
[terminal execution state](evidence/stock-utility-20261004/execution-state.md).

That original study produced no new accepted plan, consumer or transfer. It
remains closed, and its unused budget cannot be used to retry or waive the gate.
All three original implementation-review findings were fixed before collection.
The separately registered follow-up below tests acceptance under a new timing
and observability contract. The [task list](TODO.md) and
[proof and fallback plan](protocols/plans/2026-10-04-placement-proof-and-stock-fallback.md)
keep the studies and their verdicts separate.

Follow-up execution is authorized. The [bounded repeatability protocol](protocols/specs/2026-10-04-stock-repeatability.md)
passed implementation review/verification before native freeze: fixed 30-second spacing and
diagnostics, two independently passing acceptance blocks, then conditional
consumer/untouched transfer. Maximum 148 new native processes; first failed or
inconclusive gate stops. All five review findings were reproduced/fixed; 797 tests
passed, seven optional historical-source modules skipped, and six pinned native
source checks passed separately. [Execution state](evidence/stock-repeatability-20261004/execution-state.md)
records collection from `86388e7` with 41 bound files. All 148 native calls
completed: both independent main blocks, the main consumer, untouched transfer
utility/product and transfer consumer passed. Transfer gain was +9.38%, CI95
[+7.92%, +10.61%], with manual equivalence and equal 18-evaluation search budgets.
The separate raw auditor reconstructed all records and statistical gates.
Outer verdict is **PASS-STOCK-REPEATABILITY-TRANSFER**, process exit 0; complete
source/phase/receipt reconstruction passed before final publication. The
[independent final audit](evidence/stock-repeatability-20261004/final-audit.json)
matches the saved report and all 148 native records. See
[results and costs](evidence/stock-repeatability-20261004/report.md).
Fresh read-only CLI validation also passed with exit 0 and zero additional native
calls; see [final verification](evidence/stock-repeatability-20261004/native-verification.json).
Public stock CLI consolidation and source-preserving reader reuse are now
implemented separately from that bounded study. One read-only public validation
passed in 207.94 seconds, and the final adapter passed in 221.15 seconds, versus
the earlier 2,191.48 seconds. Five database readers were used, with zero extra
native calls; all 41 source/prerequisite files and six history pins remained
unchanged. These are descriptive timing comparisons. See
[implementation and archived source evidence](evidence/stock-cli-20261005/report.md).

The [wider utility comparison](evidence/stock-coverage-20261005/report.md)
completed from `78ad5f0`: **COMPLETE-STOCK-COVERAGE**, exit0, 344 retained calls,
four **NO-UTILITY-GAIN** outcomes. Q4 prose/code gained +1.03%/+0.78%, below
the fixed 5% gate. Granite selected the 8-thread/graphs-on default, with paired
changes−0.02%/−0.23%. Each case retained 86 records and equal 18/18 searches;
no new product or consumer ran. Fresh public validation passed in 93.40 seconds
with zero extra calls and unchanged original source/history evidence.
Independent final raw audit passed with no material discrepancies. Wider useful tuning, other hosts and
serving performance remain unproved; the positive Q6 scope is unchanged.

The follow-on [offload/attention/batch eligibility audit](evidence/stock-control-scope-20261006/report.md)
is complete at a **scoped no-go**, with 16 pinned upstream objects, zero native
calls and 55 passing contract tests. Offload and flash attention change numerical
paths; nonbinding batch caps have no demonstrated serial-decode benefit on
these short prompts. Further native optimization needs a new exact numerical
qualification and useful mechanism, or a separate quality policy/provider and
held-out acceptance protocol. This is the current research blocker, not a
universal impossibility result. Both completed studies and their budgets stay closed.

The [follow-through](evidence/stock-followthrough-20261006/report.md) closed the
explicit-enable flash-attention rationale at **NO-NEW-SUPPORTED-FUSED-DECODE-MECHANISM**:
pristine stock already probes AUTO support; forcing enable supplies no new
supported GPU kernel. No native quality or speed experiment followed. The fail
path produced a readable archived-status helper and delegated public verification,
with 55 passing helper/CLI/reader controls after one independent review, its fix
pass and a live registration-argument correction. The failed initial read-only
qualification is preserved. Corrected fresh local qualification from `c9d2f97`
passed both public validators in 331.65 seconds (Q6 212.65; wider 118.90), with
zero additional native calls and unchanged source/report/start digests; see the
[receipt](evidence/stock-followthrough-20261006/verification.json).
Compiler/provider sources remain unchanged. The [user handoff](stock-product-handoff.md)
is prepared. The user subsequently ran verification: Q6 passed and wider hit
the GPU environment guard. At the user's request, Luna completed the agent
interpretation walkthrough and the root agent closed the observed GPU-consuming
Zen processes and Terminal instance. A separate fresh full check then passed
in 313.51 seconds with zero extra native calls; all blocked prefixes remain
preserved in the [walkthrough archive](evidence/stock-agent-walkthrough-20261006/report.md).
This establishes agent-run local reconstruction, not five-task unaided human
success. Independent-user workflow acceptance, demand and portability remain unmeasured.

## Verified milestones

| Work | Result and scope | Evidence |
| --- | --- | --- |
| Compiler spine | Implemented typed inputs/plans, adapters, evidence store, calibration, runner and validation; original replay gate stopped | [Phase 3 checkpoint](evidence/compiler-phase3/execution-state.md) |
| Gemma Q6 stock product | Fresh paired acceptance passed; 32-run search retained 12 threads/graphs on | [Q6 stock report](evidence/stock-discovery-20261004/report.md) |
| Gemma Q4 stock product | Separate artifact/reference/product/search passed; 48 native processes; incumbent retained | [Q4 report](evidence/stock-discovery-20261004/q4-report.md) |
| Granite Q6 stock product | Real second-family reference/product passed; 68 native processes; GPU-resident incumbent retained | [Granite report](evidence/compiler-granite-20261004/report.md) |
| Gemma phase-aware profiling | CPU expert work dominated synchronized decode; diagnostic instrumentation perturbs overlap | [Profile report](evidence/compiler-phase-profile-20261004/report.md) |
| Gemma Q6 utility fallback | +12.70% over resolved thread/graph defaults; manual equivalence passed; final product equivalence inconclusive, no transfer | [Utility result and audit](evidence/stock-utility-20261004/report.md) |
| Gemma Q6 repeatability/transfer | Both main blocks and consumers passed; held-out +9.38% over resolved defaults, manual equivalence and fresh product PASS; 148 calls | [Follow-up proof](evidence/stock-repeatability-20261004/report.md) |
| Q4/Granite wider utility | Four complete NO-UTILITY-GAIN cases, 344 retained calls; public validation and independent audit PASS | [Separate results](evidence/stock-coverage-20261005/report.md) |

Gemma Q6/Q4 use pristine CPU-MoE; Granite uses a pristine GPU-resident baseline.
All three retained 12 threads and CUDA graphs on in their declared search spaces.
Granite's small-model TPS and Q4's different quantization are not Q6 speedups.
Live coverage is two families on one pinned Windows/NVIDIA host/build, not
universal model/hardware support or a global stock optimum.

Latest full suite: **891 passed, 7 historical source-environment skips**;
**6 applicable pinned native source checks and 7 installed-wheel checks passed** separately.
See [wider implementation verification](evidence/stock-coverage-20261005/implementation-review.md).
These tests establish implementation checks; live results come from the native
records and independent audits linked above.

## Closed results

- Historical Gemma static placement: **28.13 TPS**, **22.48%** above the separate
  strongest historical stock reference, but **QUALITY STOP**. PPL upper 95%
  confidence bound +2.25% exceeded +1%; cross-backend response hashes differed.
  [Placement report](evidence/q6-placement-final/report.md).
- Original Phase 3: **VALIDATION-STOP**; sealed replay drift exceeded its original
  tolerance. Fresh later stock acceptance does not change that verdict.
- Reactive/predictive caching and mover paths: no-go; no accepted cache speedup.
- Eight-thread diagnostic, Q6 16-thread challenger, CPU prefetch and PDL-off:
  no accepted improvement. [Prefetch](evidence/compiler-cpu-prefetch-20261004/report.md),
  [PDL](evidence/compiler-cuda-pdl-20261004/report.md).

## Documentation boundaries

The root README and this page describe current development. `docs/PRODUCT.md`
contains current compiler architecture followed by the historical placement
guide. Judge/deployment/submission guides describe older release paths.
`expertflow demo --replay`, dashboards, the archived release directory and ZIP
reconstruct historical evidence; replay success is not quality acceptance.
Hashed release artifacts and experimental receipts remain immutable. Inspect
Git status before staging: existing local notes/transcripts are not this plan.
