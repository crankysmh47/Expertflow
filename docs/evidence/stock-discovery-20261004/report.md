# Stock discovery: active investigation

The full user objective is a reproducible strongest-stock method for this model
and host, followed by reusable MoE/hardware support. It is not complete yet.

The [historical audit](history-audit.json) distinguishes pristine single-request
stock22.9667TPS, original server confirmation24.411/replay25.383TPS, static
placement28.13TPS with QUALITY STOP, and four-slot aggregate35.6699TPS with
nondeterministic outputs. The latter two are ineligible targets for the current
exact single-request workload. Different experiments/interfaces and incomplete
historical CPU environment pins prevent attributing the current22.67TPS control
to a specific regression. New contemporary comparisons must establish gains.

The current [host snapshot](host-environment.json) adds CPU topology, RAM modules,
OS, affinity, threading environment and active power scheme to the existing GPU/
runtime identities. The snapshot includes all visible and hidden AC/DC power
settings, so edits within the same scheme invalidate reuse. Capturing this
information does not change system settings.

Independent review at `19985dc` found two important acceptance gaps: receipt-only
host claims could be rewritten together, and scheme GUIDs omitted actual power
settings. Regression tests reproduced both gaps. Each product launch now captures
the actual host before process creation and records it in an EvidenceStore-hashed
launch artifact; receipt reconstruction checks that independent binding. Legacy
measurements remain unchanged. Fresh live collection has not started yet.

Work follows the [new complete plan](../../superpowers/plans/2026-10-04-stock-configuration-discovery.md):
fresh paired stock-product validation, quality-preserving bounded stock search,
then reusable model/runtime/host contracts and explicit cross-model live coverage.
The original compiler validation stop, thread/prefetch rejections and static
quality failure remain unchanged. No new validated plan is claimed by this audit.
