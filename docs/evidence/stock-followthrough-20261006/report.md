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

Focused controls currently pass 24 tests, including timeout diagnostic retention,
failed validator decisions and source/native-start mutation. Fresh local
qualification and independent implementation review are recorded separately
before this deliverable is marked complete.

The [user handoff](../../stock-product-handoff.md) supplies executable status and
verification commands and a five-task independent-user procedure. This is a
prepared research prototype. Actual user success, demand, portability and launch
readiness remain unmeasured; agent tests must not stand in for those observations.
