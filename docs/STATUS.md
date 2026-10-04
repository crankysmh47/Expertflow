# ExpertFlow current status

Updated 2026-10-04. Branch: `ef-v2`. Prior accepted evidence checkpoint:
`a793889`; measured Granite implementation: `7cbae5d`.

## Current decision

Prove useful, automatically selected MoE placement before expanding the product.
If placement cannot qualify within its frozen budget, evaluate stock autotuning
as the fallback. Both paths must demonstrate practical value. The current
supported result is validated stock selection/reproduction; an accepted new
quality-preserving speedup over tuned stock has not been demonstrated.

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

No new accepted plan, consumer run or held-out transfer was produced. Current
product scope stays validated configuration selection/reproduction, with the
narrow defaults-tuning gain reported separately. CLI expansion and broader
features are deferred. Next is a separately reviewed bounded repeatability/
acceptance protocol with a new justification; this closed study cannot be
resampled or have its gate waived. The [task list](TODO.md) and
[proof and fallback plan](superpowers/plans/2026-10-04-placement-proof-and-stock-fallback.md)
record completed work and blocked stages. All three fresh implementation-review
findings were reproduced and fixed before collection at `356b660`.

## Verified milestones

| Work | Result and scope | Evidence |
| --- | --- | --- |
| Compiler spine | Implemented typed inputs/plans, adapters, evidence store, calibration, runner and validation; original replay gate stopped | [Phase 3 checkpoint](evidence/compiler-phase3/execution-state.md) |
| Gemma Q6 stock product | Fresh paired acceptance passed; 32-run search retained 12 threads/graphs on | [Q6 stock report](evidence/stock-discovery-20261004/report.md) |
| Gemma Q4 stock product | Separate artifact/reference/product/search passed; 48 native processes; incumbent retained | [Q4 report](evidence/stock-discovery-20261004/q4-report.md) |
| Granite Q6 stock product | Real second-family reference/product passed; 68 native processes; GPU-resident incumbent retained | [Granite report](evidence/compiler-granite-20261004/report.md) |
| Gemma phase-aware profiling | CPU expert work dominated synchronized decode; diagnostic instrumentation perturbs overlap | [Profile report](evidence/compiler-phase-profile-20261004/report.md) |
| Gemma Q6 utility fallback | +12.70% over resolved thread/graph defaults; manual equivalence passed; final product equivalence inconclusive, no transfer | [Utility result and audit](evidence/stock-utility-20261004/report.md) |

Gemma Q6/Q4 use pristine CPU-MoE; Granite uses a pristine GPU-resident baseline.
All three retained 12 threads and CUDA graphs on in their declared search spaces.
Granite's small-model TPS and Q4's different quantization are not Q6 speedups.
Live coverage is two families on one pinned Windows/NVIDIA host/build, not
universal model/hardware support or a global stock optimum.

Latest full suite: **767 passed, 7 historical source-environment skips**;
**6 applicable pinned native source checks passed** separately.
See [utility implementation verification](evidence/stock-utility-20261004/implementation-verification.json).
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
