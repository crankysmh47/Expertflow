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
measurements remain unchanged. Fixes were committed at `a527d9b` after
538 tests passed (7 skipped).

Work follows the [new complete plan](../../superpowers/plans/2026-10-04-stock-configuration-discovery.md):
fresh paired stock-product validation, quality-preserving bounded stock search,
then reusable model/runtime/host contracts and explicit cross-model live coverage.
The original compiler validation stop, thread/prefetch rejections and static
quality failure remain unchanged.

## Fresh paired product validation

The fixed twenty-process product experiment completed at source `a527d9b` with
`PASS-STOCK-FALLBACK`. Direct mean22.792150TPS; sealed mean22.922946TPS;
geometric difference+0.569334%, CI90[-0.048785%,+1.340202%], both within the
frozen2% equivalence margin. CVs0.739741%/1.258123%. All twenty native records
passed exact prompt/generated tokens, owned memory/reserve and cleanup, with
unique owned process identities. No retries or discarded samples.

The [independent audit](product-verification.json) reverified every record,
recomputed the bootstrap and confirmed original plan/database unchanged. This
is replay/product acceptance, not an optimization or historical speed recovery.
Published plan SHA256:
`3849427ac69fdab14babedb00d0a3b3fd4c0a44420a7b8aa32963fe805cc3497`.

Live evidence database:
`C:/models/expertflow/runs/compiler-stock-product-20261004/compiler.sqlite3`.
Published [plan](accepted/execution-plan.json) and
[receipt](accepted/acceptance-receipt.json) are preserved here and in that run's
`product/accepted/`; the report and frozen
protocol are in `product/`. Use the existing CLI `validate --acceptance` with
these files and pinned compiler inputs to reconstruct acceptance without a
new native run. The live CLI returned `VALIDATED-STOCK-FALLBACK` with the same
plan hash and no extra native process. Original Phase3 and older A/A artifacts
retain their verdicts.

Next work follows the [bounded search plan](../../superpowers/plans/2026-10-04-bounded-stock-search.md).
Generic candidate generation, fixed complete-block screening and temporal
normalization/ranking passed25 synthetic contract tests after observed RED.
No stock search native samples or cross-model results have been collected.

The reviewed numerical eligibility provider is implemented for the pinned Gemma4
Q6 artifact and pristine runtime only. Its nine additional tests reject changed
weights/build/source objects, unsupported architecture/family/quantization and
misbound provider identities;34 search/eligibility tests pass together. The
[live input attestation](scheduling-eligibility.json) verified actual pinned
runtime binaries and immutable source objects without launching a native model.
A synthetic second-family provider tests the extension contract only. Q4 weights
exist locally, but Q4 and other families remain ineligible until their own
operation-path proof and validation are implemented.
