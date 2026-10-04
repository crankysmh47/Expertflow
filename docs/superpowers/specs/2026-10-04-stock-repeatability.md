# Bounded stock repeatability follow-up

User authorized proceeding with the next bounded repeatability protocol after
the [closed utility study](../../evidence/stock-utility-20261004/report.md).
This is a new timing/observability contract and stricter replication test;
the original 106-process PRODUCT-VALIDATION-STOP remains unchanged.

## Question and justification

Can the fixed, previously selected Gemma Q6 configuration reproduce direct/
sealed equivalence in two fresh blocks with uniform pre-launch spacing?
Retained native arguments, requests and candidate identities match; sealed
preparation is under 0.6ms. The old acceptance had no CPU/GPU diagnostic samples,
so thermal/background-load causation is unknown. The new protocol supplies
diagnostics and fixes elapsed pre-launch spacing at 30 seconds for every arm.
It tests repeatability under that contract, not a proven thermal repair.
No larger sample pool, favorable subset, new candidate or relaxed margin is used.

Alternatives considered: simply add samples (rejected); change the statistical
gate to noninferiority only (rejected); stop at validation/reproduction forever
(retained fallback if this bounded attempt fails). Two separately evaluated
blocks protect against treating one favorable repeat as resolution.

## Fixed scope and invariants

- Same pinned Gemma Q6 bytes, pristine a7312ae CUDA runtime, Windows/NVIDIA host,
  exact native tokens, F16 KV, CPU-MoE/ngl99, context4096/predict512/seed42,
  concurrency1/batch2048/ub512. No native source/build or compiler-library change.
- Main candidate is the old utility-selected 12 threads/graphs on. Its structural
  plan is diagnostic input until new acceptance; the old failed product is never
  relabeled accepted. Reverify its ten independent confirmation records and the
  original utility PASS, source/host/manifest and raw artifacts before collection.
- Reuse the committed main and untouched code prompts. Freeze both loaded input
  snapshots, the selected plan/database identities, all collector/compiler/
  diagnostic source hashes, protocol, host and complete budget before launch.
- Every native call waits at least 30 seconds before invoking the owned runner,
  including the first call, consumer, transfer search and transfer product.
  This is fixed spacing, not a claim of measured CPU idle or a temperature target.
  Source/host guards run before and after the wait. No adaptive cooling or retry.
- Use existing DiagnosticSampler at the runner's 0.2-second memory interval for
  all calls. Optional sensor unavailability is retained; mandatory owned memory,
  reserve and cleanup remain fail-closed. CPU temperature is unavailable.
- Pre-launch waiting, native phase time and full collector elapsed time are
  separate costs. Do not claim lower wall time or general deployment throughput.

## Sequence, budget and stop rules

| Phase | New native maximum | Exit requirement |
| --- | --- | --- |
| Main acceptance block A | 20 | Ten fresh balanced direct/sealed pairs PASS |
| Main acceptance block B, conditional on A | 20 | Another ten fresh pairs PASS independently |
| Main consumer, conditional on both | 1 | Fresh accepted-plan exact execution PASS |
| Untouched transfer, conditional on consumer | 107 | Own reference10, automatic18, manual18, default pairs20, manual pairs20, product20, consumer1 |
| Total | 148 | Every activated phase and independent reconstruction PASS |

Each main block uses the unchanged paired-stock-product-v1 protocol:
seed20261003, 10,000 whole-pair nearest-rank bootstrap draws, balanced schedule,
CI90 strictly inside ±2%, one-sided95 lower>-2%, each arm CV<=10%, native token/
identity/memory/ownership/cleanup PASS. Evaluate blocks separately; both must
pass. No pooled result can rescue an inconclusive block. Publish the new main
artifact only after both blocks pass; use block B's standard product receipt,
and retain both reports in the encompassing repeatability result.

Transfer reuses the unchanged stock utility algorithm and gates under this
new protocol's fixed spacing/diagnostics. This is a newly registered conditional
budget, not activation of the closed study's unused budget. The original
106-process verdict stands. The prior main utility PASS plus the new two-block
acceptance/consumer gate is the prerequisite for this transfer. The held-out
prompt was committed before the original collection and remains unmeasured.
Freeze all phases now; no source/ranking/search/control changes between them.

Transfer requires >=5% geometric gain over resolved8/on defaults, positive
CI95 lower, manual CI90 within±2%, CV<=10%, equal18 native search evaluations,
then standard fresh product equivalence20 and consumer1. Neutral/default-optimal
cases stop the utility claim. Optional confirmation samples cannot be added.

Persist every attempt before launch and observed native identity on exceptions.
Maximum148 attempts; the nested transfer maximum107 also applies. Failures,
source/host drift, missing native evidence, timeout or inconclusive statistical
gate immediately end the study. No ordinary-sample discard, pilot, retry,
alternate finalist, gate change or third repeatability block. No accepted root
artifact before both blocks pass; a later consumer/transfer failure remains
explicit even if narrow main acceptance passed.

## Implementation and verification

Create `scripts/benchmark_compiler_stock_repeatability.py` as an orchestration
wrapper over existing execute_pairs, publish_stock_product, accepted consumer
and execute_utility. Keep their source files and historical protocols unchanged.
Its GuardedRunner owns the outer freeze, fixed waiting and durable attempt count;
the existing transfer FrozenRunner retains the nested utility manifest/context.
Require fresh database/output paths; validate is read-only and launches no model.

Test before implementation in `tests/test_compiler_stock_repeatability.py`:
both-pass publication, first/second block stops, consumer/transfer conditional
entry, wait/guard ordering, cross-phase source drift, retained post-launch
exceptions, root and148 budget rejection, historical prerequisite rejection,
and reconstructed receipts/statistics/context. Use existing native artifact
fixtures; never substitute fake PASS booleans for production gates.

Run focused tests, one full post-review suite, applicable pinned source checks,
compileall and diff checks. One fresh independent final reviewer checks this
protocol and complete implementation before the first native freeze. Reproduce
and fix material findings in one pass, then freeze. Independently reconstruct
new raw native records, diagnostics availability, delays, costs and statistics
before public claims. Preserve original reports/source hashes and unrelated files.

## Result and product scope

If both blocks and consumer pass, main repeatability is qualified under this
timing/diagnostics contract; the old study is still stopped. Transfer determines
additional workload utility coverage. There is no custom placement speedup,
global optimum, universal gain, proven thermal cause or unattended CLI performance
claim. Broader CLI/features remain outside this protocol. Failure preserves the
narrow default-tuning result and validated selection/reproduction product scope.

## Recorded invocation

Run from the reviewed `ef-v2` source checkpoint only after implementation/tests/
independent review. The directory must not exist; no resume/retry is permitted.

```powershell
uv run --no-sync python scripts/benchmark_compiler_stock_repeatability.py --action run --output-dir C:/models/expertflow/runs/compiler-stock-repeatability-20261004
```

Reconstruct without launching a model, using the same command and `--action
validate`. It returns nonzero for a terminal negative result. Default arguments
bind the original Q6 descriptor/inventory/hardware/runtime, committed main/code
workloads, original selected-plan/source database/report and pinned native
repository. Supplied arguments must pass identical scope/prerequisite checks.
