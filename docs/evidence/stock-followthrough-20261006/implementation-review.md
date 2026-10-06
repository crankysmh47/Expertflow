# Independent whole-change review and one fix pass

Reviewer: `stock_followthrough_review`, fresh context, read-only inspection and
focused CPU controls. No model launches, live qualification or full suite.
Initial review: two Important findings, no Critical findings, one Minor omission.

- Important: the catalogue's report digests had no independent catalogue pin.
  Altering both could display invented statistics. A reproducing control failed
  before the fix. The helper now requires the catalogue's independently stored
  SHA-256 before reading any report; changing both catalogue and report stops.
- Important: malformed nested `studies`/`decision` containers escaped as
  `AttributeError`, and validator failure lacked a verification summary. Both
  reproducing controls failed before the fix. Explicit object checks now return
  identity-stop or retain `verification.json` with unverified status and logs.
- Minor presentation requirement, fixed in the same pass: plain output omitted
  native counts, manual intervals and default selection; Q6 JSON omitted default
  selection. The reproducing presentation control failed, then passed after all
  five rows included these fields and the readable output included both counts.

All four controls were observed RED. Final helper suite: **28 passed** in 6.26
seconds. Broader helper/public CLI/reader controls: **54 passed** in 125.98 seconds,
recorded in `final-focused-tests.log`. Live qualification is recorded separately.
No second review or independent re-review pass is claimed.

The reviewer verified all three archived upstream objects, the frozen launcher,
492 launch/run-start digests and 41/63 source bindings. The source feasibility
interpretation is sound only for its stated explicit-enable rationale: AUTO
already probes support; forcing enable supplies no new supported GPU kernel.
Effective per-layer modes, native quality and performance remain unmeasured.

Subsequent live integration exposed a separate argument-shape defect: the wider
manifest's registration is embedded JSON rather than a filename. Windows 206
stopped the wider subprocess after a passing Q6 validator. This is preserved as
a failed qualification, not hidden by the CPU/review checks. The registration
argument regression was observed RED; the fixed project-file binding then
passed all **55 focused tests** in 134.24 seconds. No independent re-review is
claimed for this one-line integration correction.
