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
The search's terminal native result and independent audit are recorded below.

The reviewed numerical eligibility provider is implemented for the pinned Gemma4
Q6 artifact and pristine runtime only. Its nine additional tests reject changed
weights/build/source objects, unsupported architecture/family/quantization and
misbound provider identities;34 search/eligibility tests pass together. The
[live input attestation](scheduling-eligibility.json) verified actual pinned
runtime binaries and immutable source objects without launching a native model.
A synthetic second-family provider tests the extension contract only. Q4 weights
exist locally, but Q4 and other families remain ineligible until their own
operation-path proof and validation are implemented.

## Bounded Q6 stock search: retain incumbent

The fresh search at `f22bb54` completed exactly32 owned native processes: twelve
screening runs across threads12/16 and CUDA graphs on/off, then twenty balanced
confirmation runs for the screening finalist, threads16/graphs on. Screening
did not establish a gain. Independent confirmation found the challenger slower:
incumbent22.715732TPS, challenger22.492510TPS, geometric change-0.985015%,
CI95[-1.383026%,-0.551075%], CVs0.419660%/0.836433%. It failed the frozen
gain>=2% and CI95 lower>0 gates. No retries, discarded runs or alternate finalist.

Terminal status is `RECOMMENDED-INCUMBENT`: threads12, CUDA graphs on, pristine
CPU-MoE/ngl99, unchanged Q6 bytes and F16KV/workload. The existing accepted plan
hash remains `3849427ac69fdab14babedb00d0a3b3fd4c0a44420a7b8aa32963fe805cc3497`.
The [published recommendation](recommended/execution-plan.json) and
[search receipt](recommended/search-receipt.json) preserve the negative
confirmation and complete space coverage; retaining an incumbent is not a new
speedup. [Independent verification](search-verification.json) rechecked all32
native artifact sets and unique owners, reconstructed ranking/bootstrap/gates,
confirmed frozen source and prerequisite files unchanged and validated the plan.

Coverage is four declared configurations, not a global optimum. Threads8 was
explicitly excluded by the prior rejected hypothesis; its old rate was not
reused as a current sample. All other thread counts and other stock controls
remain untested/ineligible without their separate numerical contracts. This
experiment's throughput is close to historical CLI stock22.9667TPS; that is
descriptive, not a matched comparison. It does not reproduce the older server
24.411/25.383TPS, explain interexperiment drift or make the quality-ineligible
28.13TPS eligible. The original replay and static no-go verdicts remain intact.

Database: `C:/models/expertflow/runs/compiler-stock-search-20261004/compiler.sqlite3`.
Raw/frozen/report artifacts: the sibling `search/` directory. A clean checkout
of the measured revision is preserved at
`C:/models/expertflow/worktrees/compiler-stock-search-f22bb54`; see the
[method guide](../../stock-configuration-method.md) for exact validation commands
and source-preservation rules. Q4 normalization/provider/live validation and
accepted recommendation execution remain the next Stage C work.

## Stage C terminal update

The separately reviewed reuse implementation and Q4 validation are complete.
Q4's actual inventory/weights and numerical scope were verified, ten fresh
reference processes were stable, twenty paired product processes passed, and
all eighteen screening runs ranked the accepted12-thread/graphs-on incumbent
first. No self-confirmation was required. Its audits and actual CLI validators
passed; see [Q4 report](q4-report.md). This is a separate quantization, not a
Q6 quality-preserving speedup or second-family live result.

Trusted model/provider extension contracts, multiple synthetic inventories and
topologies, identity invalidation and accepted search execution/CLI are tested.
Post-review suite passed670 tests with7 source-environment skips; six pinned
source checks passed separately. Accepted search consumer uses native artifact
fixtures, with no extra native run added to either registered search budget.
Final [method verification](method-verification.json) reverified all100 distinct
registered native records and source revisions. The [objective audit](objective-audit.md)
preserves remaining coverage limits and the unrecovered historical server rates.
