# Bounded stock repeatability execution state

Registered [protocol](../../superpowers/specs/2026-10-04-stock-repeatability.md).
Branch `ef-v2`; original study remains PRODUCT-VALIDATION-STOP.

Current state: implementation verified, fresh native freeze next. The full
post-review suite passed 797 tests, with seven optional historical-source modules
skipped; six applicable pinned source checks passed separately. Thirty focused
tests and the final persisted-PASS regression passed. All five Important review
findings were fixed in one tested pass.
No new native inference process or scientific result at this checkpoint.

Raw output, once verification permits the first freeze:
`C:/models/expertflow/runs/compiler-stock-repeatability-20261004`.
Fresh directory only; no resume, retry, discarded sample or extra candidate.
The first failed/inconclusive gate ends collection.

Maximum 148 new native calls: main A20, independently qualified B20, consumer1,
then conditional untouched transfer107. Every call waits at least 30 seconds.
The existing paired equivalence, correctness, variance and utility gates remain
unchanged. Source/host guards and separate attempt journals cover all phases.

Native performance, optional diagnostic availability and transfer are unmeasured.
Fixed waiting is a timing contract; thermal/background-load causation is unknown.
Broader deployment throughput and CLI expansion remain outside this study.

See the [read-only diagnosis](diagnosis.json), [prerequisite preflight](preflight.json),
[fresh implementation review](implementation-review.md) and
[independent raw auditor](independent_audit.py).
