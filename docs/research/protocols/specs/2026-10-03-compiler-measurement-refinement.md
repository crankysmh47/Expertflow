# Compiler measurement refinement protocol

User authorization: follow the recommendation to refine measurement, then test
one focused optimization. Work on ef-v2; preserve the Phase 3 VALIDATION-STOP.
This bounded experiment extends the existing owned runner and evidence store.

## Frozen question and inputs

Does loading and executing the same stock settings from a sealed plan preserve
performance compared with direct stock settings? This A/A comparison cannot
demonstrate an ExpertFlow optimization gain. Use the selected diagnostic pending
plan from docs/evidence/compiler-phase3/execution-plan.pending.json, validated
against its original evidence DB; do not reclassify that artifact as published.
Use the original pinned Q6 weights, pristine runtime, hardware snapshot and exact
context4096 / predict512 / threads12 / seed42 / F16 KV / graphs-on workload.
Both arms use ngl99 and cpu_moe=true. Direct constructs RuntimeSettings from those
frozen values; sealed reloads the exact candidate. Both use the same owned native
HTTP transport, complete token/memory/cleanup checks and fresh OS processes.

## Measurement budget and ordering

Exactly ten pairs, twenty retained cold processes, no retries or discarded
outliers. Five pairs are direct-first and five sealed-first, shuffled once with
seed20261003. Preserve the realized schedule before launching. A failure stops
the run and retains partial evidence. A fresh database/output directory owns this
experiment; original raw records and database remain unchanged. Freeze source
commit, protocol hash, selected plan hash and all inputs before collecting data.

Measure native predicted-token TPS and whole completion wall time separately.
Keep startup/plan-loading wall time diagnostic. Sample owned GPU memory every
200ms using existing fail-closed PDH/NVML checks. Capture GPU SM/memory clocks,
temperature, power and utilization, CPU performance/frequency counters and owned
CPU time within the same sampling call. Missing diagnostic sensors are explicit
unavailable values, never invented zeroes; they do not weaken mandatory memory
checks. CPU temperature is unavailable unless a trustworthy sensor exists.
Record actual offloaded layer count when logs expose it; otherwise say unknown
and use byte-measured owned allocation as a diagnostic, never claim deduplication.
No synchronized backend profiling is enabled in retained timing runs.

## Statistics and decision rules

For each pair calculate log(sealed_TPS/direct_TPS). Resample whole pairs with
replacement 10,000 times using seed20261003. Convert bootstrap means through
100*expm1. Report the point geometric-mean percentage change, two-sided95%
percentile CI, one-sided95% lower bound and two-sided90% CI. Preserve every rate
and per-arm CV. This is a small-sample percentile-bootstrap estimate, not a
distribution-free guarantee; temporal dependence remains a limitation.

Correctness requires identical model/hardware/runtime/workload/settings,
independent owned-run identities, and identical prompt/generated native token
hashes across both arms and all pairs. Memory, reserve, telemetry and cleanup
retain the original strict validations. Invalid or missing evidence fails closed.

Deployment non-inferiority passes when the one-sided95% lower bound is above
-2%. Repeatability equivalence requires the entire two-sided90% CI within
[-2%,+2%] (the interval form of two one-sided5% tests). Both arm CVs must be <=10%.
PASS-MEASUREMENT requires all correctness checks, non-inferiority, equivalence and
CV checks. A confidence interval crossing a boundary is INCONCLUSIVE. Clear
regression is VALIDATION-STOP. Faster A/A is not called an optimization gain.
Neither result rewrites the original absolute2% verdict or publishes a product
plan automatically. No optional stopping or widened margins after seeing data.

## Conditional next experiment

Only after PASS-MEASUREMENT, use this experiment's CPU/GPU diagnostics and the
existing synchronized backend profiles to identify a bottleneck. On this Ryzen7
9700X (8 physical cores / 16 logical), a candidate hypothesis is that twelve
threads contend while CPU expert work dominates. Select exactly one candidate,
eight threads versus the frozen twelve-thread stock baseline, only if evidence
supports CPU work as the primary target. Before launching, freeze a separate
protocol and verify the pinned CPU kernel's threading partition preserves each
output's reduction path. If that mechanism cannot be established, record a
numerical-contract blocker rather than label finite token parity as bitwise proof.

Use ten balanced pairs with the same fixed-budget statistical method. Retain
stock runtime, KV, numerical kernels, GPU placement, prompt and token budget.
The acceptance target is point gain >=5%, two-sided95% lower bound >0, exact
tokens, memory and cleanup. Failure/inconclusive ends this hypothesis; no sweeps,
dynamic caching, approximate placement or new compiler architecture in this work.

## Implementation and verification

Add focused paired statistics/orchestration in src/expertflow/compiler/refinement.py,
optional read-only sensor sampling in src/expertflow/compiler/diagnostics.py and a
thin scripts/benchmark_compiler_refinement.py entry point. Preserve public Phase3
schemas and old compile/replay behavior. Regression tests cover improved and
regressed distributions, uncertain equivalence, missing/duplicate owned runs,
unstable tokens, balanced schedule, frozen candidate binding and partial failures.
Use the existing real HTTP-child integration test for sensor failure behavior.
Run the full CPU suite and compileall before native work; make one fresh final
code review and one fix pass if needed. Save protocol, input/source hashes,
schedule, raw artifact manifest, paired rates, telemetry summary, uncertainty,
commands/exit codes and explicit verdict under docs/evidence/compiler-refinement.
Append only this work's project-log entry; preserve the pre-existing R1 entry.

References: paired resampling semantics in
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html;
interleaving rationale in https://google.github.io/benchmark/random_interleaving.html;
sensor APIs in https://docs.nvidia.com/deploy/nvml-api/latest/index.html.
