# Bounded stock repeatability execution state

Registered [protocol](../../superpowers/specs/2026-10-04-stock-repeatability.md).
Branch `ef-v2`; original study remains PRODUCT-VALIDATION-STOP.

Terminal state: PASS-STOCK-REPEATABILITY-TRANSFER, process exit 0. All 148 native calls completed; the
untouched transfer passed utility, fresh product20 and consumer1. Its automatic
and manual grids each used 18 evaluations and selected 12 threads/graphs on.
The [final independent raw audit](final-audit.json) reconstructed all 148 records
and their statistical gates, matching the final report SHA. The frozen outer
source/phase/receipt reconstruction passed before final PASS. Separate fresh
read-only CLI validation passed with exit 0, zero additional native calls and
2191.484 seconds wall time. See [results and costs](report.md) and
[final verification](native-verification.json).
Both independent 20-call main blocks A/B
passed their original acceptance gates; the main artifact was published under
`block-b/accepted` only after both passed. The fresh main consumer passed exact
execution, and the conditional transfer study has started with its own reference.
Transfer utility/product/consumer and final outer gates passed.
Native calls ran October 5 locally
from source `86388e7529b1fa8fa927e2baebc3f5ce84c6d597`; the output path retains
its October 4 registration date. The [outer manifest](main-frozen-manifest.json)
binds 38 collector/protocol sources plus three original prerequisite artifacts.
The full
post-review suite passed 797 tests, with seven optional historical-source modules
skipped; six applicable pinned source checks passed separately. Thirty focused
tests and the final persisted-PASS regression passed. All five Important review
findings were fixed in one tested pass.
Native collection, outer reconstruction and fresh read-only verification ended
successfully. No model-server processes remain. Final documentation and evidence
are recorded on the active `ef-v2` branch; source41/history6 remain unchanged.

Raw output, once verification permits the first freeze:
`C:/models/expertflow/runs/compiler-stock-repeatability-20261004`.
Fresh directory only; no resume, retry, discarded sample or extra candidate.
The first failed/inconclusive gate ends collection.

Maximum 148 new native calls: main A20, independently qualified B20, consumer1,
then conditional untouched transfer 107. Every call waits at least 30 seconds.
The existing paired equivalence, correctness, variance and utility gates remain
unchanged. Source/host guards and separate attempt journals cover all phases.

Bounded repeatability and transfer qualification passed; wider product scope remains unverified.
Fixed waiting is a timing contract; thermal/background-load causation is unknown.
Broader deployment throughput and CLI expansion remain outside this study.

See the [read-only diagnosis](diagnosis.json), [prerequisite preflight](preflight.json),
[fresh implementation review](implementation-review.md) and
[independent raw auditor](independent_audit.py).
The auditor now covers the held-out ranking, paired utility/product gates and
native phase costs, with [positive, negative and tamper controls](audit_controls.py).
It is outside the frozen collector sources; the running protocol is unchanged.
