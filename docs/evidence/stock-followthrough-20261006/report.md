# Research gate and bounded stock prototype continuation

The explicit-enable flash-attention candidate closes at
**NO-NEW-SUPPORTED-FUSED-DECODE-MECHANISM**. The fail path continues into the
bounded stock prototype; it does not reopen placement, tune another attention
mode or create an approximate provider. No native quality or speed experiment
was performed for this candidate.

## Source feasibility

The [gate reconstruction](research_gate.py), [machine result](research-gate.json),
[source inventory](research-source-inventory.json) and
[complete source archive](research-source-objects.zip) bind pristine upstream
`a7312ae94f801fc9c6786dc56e38df57b964f697` and the measured local launcher.
All 492 existing launch records were checked for explicit flash-attention flags;
the launcher filters inherited `LLAMA_ARG_` overrides. This establishes the
declared AUTO default, not the effective per-layer mode in a native trace.

`common/common.h:496` defaults to AUTO; `common/common.cpp` forwards that setting.
`llama-context.cpp:228-229,473-525` starts with fusion enabled and resolves support
by backend placement. If the probe accepts, AUTO already enables the fused path.
If it rejects, explicit enable supplies no additional supported GPU kernel.
Thus this proposed rationale does not justify a new useful control. Initialization
probe costs are outside the declared decode objective. The audit does not
prove every attention setting useless or establish measured quality loss.

Three upstream objects and the bound launcher are preserved. Source and
run-start/report digests are unchanged; **zero additional model processes**.
The historical placement, prefetch, PDL, utility and wider verdicts remain closed.

## Continued product path

The [checkout helper](../../../scripts/stock_product_status.py) reads the
[pinned catalogue](catalogue.json), presents one Q6 held-out result and four
separate wider outcomes, and distinguishes archive inspection from fresh
reconstruction. It delegates qualification only to the consolidated public CLI.
Missing or changed inputs, nonfinite intervals, invalid statuses, a mismatched
report identity or mutated source/run-start inventory cannot produce a fresh
success claim. Protected study roots and occupied outputs are rejected.

One independent implementation review found two Important defects and one
presentation omission; all four reproducing controls failed before the fix pass.
Catalogue byte pinning, nested JSON handling and readable comparator/default
details were corrected. Initial final verification passed 54 focused controls.

The first live read-only qualification at `c41672e` retained a
`VALIDATION-STOP`: Q6 public validation passed in 222.84 seconds, then Windows
error 206 prevented the wider validator from starting because an embedded
registration object was incorrectly passed as a filename. The complete
[failed attempt](precorrection/verification.json),
[preservation record](precorrection/preservation.json) and original helper source
snapshot are retained. Zero model calls; original 41 files/six pins/148 starts
and wider 63 files/344 starts remained unchanged.

A reproducing registration-argument control failed before the one-line binding
fix. Corrected focused verification: **55 passed**, including 29 helper controls,
in 134.24 seconds. The corrected helper binds the fixed project registration
file; the existing public CLI attests it.

Corrected live qualification from `c9d2f9700e59d45eac47d92ebd64c1ce98717deb`
passed **PASS-LOCAL-STOCK-QUALIFICATION** in 331.65 seconds, with **zero
additional native calls**. Q6 reconstruction passed in 212.65 seconds using
five readers/model-cache entries; wider reconstruction passed in 118.90 seconds
using four. Both public validators exited 0 and verified their expected report
identities. These are read-only reconstruction costs, not inference speedups.
Before/after report, frozen-source and native-start inventory digests match.

The [verification receipt](verification.json), [Q6 output](q6-stdout.json),
[wider output](wider-stdout.json) and empty stderr logs are byte-identical copies
of the separate corrected output directory. The [publication manifest](publication.json)
binds their hashes and measured helper commit; the
[raw helper snapshot](corrected-helper-source-snapshot.json) binds the helper,
tests and catalogue. The [publication verifier](verify_publication.py) passed
[source and qualification bindings](publication-verification.json), including
the unchanged original 41 files/six pins/148 starts and wider 344 starts.
The initial failed attempt remains preserved; total helper overhead across both
attempts was 554.51 seconds, without new model outputs.

The [user handoff](../../stock-product-handoff.md) supplies executable status and
verification commands and a five-task independent-user procedure. This is a
locally qualified research prototype. The initial user reply was **"Not tried yet"**.
The subsequent user validation passed Q6 and stopped at the wider GPU guard;
the user requested a [Luna agent walkthrough](../stock-agent-walkthrough-20261006/report.md)
of the remaining questions. The [current usability state](usability-state.json)
records that partial attempt and assistance request, without human task timings
or five-task unaided success. Independent-user acceptance still requires an
intended user to complete the five tasks unaided.
On pass, continue scoped packaging/onboarding; on fail, fix the observed workflow
problem and repeat with a fresh user session. Actual user success, demand,
portability and launch readiness remain unmeasured; agent tests must not stand
in for those observations.
