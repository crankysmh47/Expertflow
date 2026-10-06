# Local product pilot task sheet

Protocol prepared 2026-10-06; no participant results have been collected.
Audience: local-model enthusiasts/developers; initial release Windows/NVIDIA.
Exact outputs by default; TUI conditional on observed CLI friction.

Use the same local GGUF and available runtime files for ExpertFlow and the
participant's existing/upstream workflow. Record download and dependency setup
time separately from the 15-minute install-to-response target. Do not coach the
participant during the task; record any help as a failed unaided task.

1. Install the released wheel in a clean environment and locate help.
2. Inspect their own model, identify requested context and fit uncertainty.
3. Create a baseline profile and generate a response.
4. Start serving and connect an existing client to the displayed local endpoint.
5. Run a bounded benchmark/tuning job and explain gain/no-gain/inconclusive.
6. Find the report and recover from an occupied port or failed model load.

Record anonymous participant role, exact platform/runtime/model, task success,
elapsed time, command exits, confusion, any assistance, and report digests.
Also record existing-workflow time/errors and which workflow they prefer and why.
Exclude names, private prompts, credentials and identifying local paths.

Recruit five people through human outreach. Target four completing install to
response unaided within 15 minutes when files are available, three using their
own setups, and three returning voluntarily within two weeks. These small-pilot
gates show initial usability/interest, not representative market demand.
At least two struggling with CLI navigation or report interpretation qualifies
the optional TUI investigation. Preserve all failures; repair observed issues
and repeat the affected task with a fresh session.

Agent demonstrations and fake runtimes cannot be entered as participants.
The earlier five-task evidence handoff remains a separate check.
