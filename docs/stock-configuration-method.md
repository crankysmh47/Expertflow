# Finding a validated stock MoE configuration

Current priorities and accepted scope are in [STATUS.md](STATUS.md),
[tasks](TODO.md), and the [proof/fallback plan](superpowers/plans/2026-10-04-placement-proof-and-stock-fallback.md).
This method provides validated stock selection/reproduction plus bounded
autotuning utility on the pinned Q6 workloads. Its earlier incumbent-retention
results establish reliable reproduction; utility over defaults is a separately
measured claim, with no new gain over already tuned stock.

After the scoped placement feasibility no-go, the
[bounded utility protocol](evidence/stock-utility-20261004/protocol.md) compares
automatic tuning with resolved thread/graph defaults and an independent manual
grid. The [completed result](evidence/stock-utility-20261004/report.md) confirmed
a 12.70% gain over those defaults and manual equivalence at equal evaluation
budgets. Final sealed-plan acceptance was inconclusive, so no new accepted
artifact, consumer or held-out run followed. The utility study is closed;
the earlier accepted stock plans below remain separate authorities.

The separate [repeatability and transfer proof](evidence/stock-repeatability-20261004/report.md)
passed all 148 native calls and final source/phase/receipt reconstruction. Both
main acceptance blocks and fresh consumers passed. The unchanged method applied
to an untouched code workload selected 12 threads/graphs on, gained 9.38% over
resolved 8-thread/graphs-on defaults, and matched an independent manual grid at
equal 18-evaluation budgets. Fresh held-out product acceptance also passed.
This result is qualified under fixed 30-second spacing and diagnostics, on one
host/runtime/artifact; it does not establish universal or serving performance.

Reconstruct the completed study without launching a model, from its measured
source snapshot or unchanged frozen source files:

```powershell
uv run --no-sync expertflow stock repeatability validate --output-dir C:/models/expertflow/runs/compiler-stock-repeatability-20261004
uv run --no-sync python docs/evidence/stock-repeatability-20261004/independent_audit.py --report C:/models/expertflow/runs/compiler-stock-repeatability-20261004/report.json --output C:/models/expertflow/runs/repeatability-readonly-audit.json
```

The completed 148-call budget is closed. These are read-only commands, not
authorization to rerun collection. Original source/artifact paths and local
databases must remain available; archive snapshots do not relocate their
identity bindings. Public validation reuses one verified reader per database,
with the original stat guard and repeated artifact/runtime/record checks. One
read-only validation passed in 207.94 seconds, compared with the earlier
2,191.48-second legacy invocation. The final reporting adapter also passed in
221.15 seconds. Five database readers were reused, with
zero new native calls and all 41 frozen files/six historical pins unchanged.
The exact measured sources and prerequisites are retained in the
[source archive and verification](evidence/stock-cli-20261005/report.md).
The original script stays unchanged and retains its original read costs.

## Public stock CLI

`expertflow stock [--project PATH] WORKFLOW ACTION [driver flags]` is the common
entry point. Relative input/output paths resolve in the supplied research
checkout. Installed compiler bytes must match that checkout's declared source;
help works outside it. Local GGUFs, binary/runtime/source-contract prerequisites
and evidence databases remain external. This is a research workflow, with the
same family/quantization/numerical and original path guards as its drivers.

| Workflow | Actions | Purpose |
| --- | --- | --- |
| reference | generate, run, validate | Prepare/collect/reconstruct a fresh stock reference |
| product | run | Fresh ten-pair product acceptance from a verified source plan |
| search | generate, run, validate, execute | Bounded selection, accepted recommendation and fresh execution |
| utility | validate | Reconstruct the closed original Q6 utility study |
| repeatability | validate | Reconstruct the closed 148-call follow-up with reader reuse |
| coverage | inspect | Verify wider registration metadata; never launch a model |

Use workflow/action help for the original input flags, for example:

```powershell
uv run --no-sync expertflow stock reference generate --help
uv run --no-sync expertflow stock product run --help
uv run --no-sync expertflow stock search generate --help
uv run --no-sync expertflow stock coverage inspect
```

Product collection always selects the stock-product experiment; action/experiment
override flags and their abbreviations are rejected. Consume an accepted
product using the existing public `expertflow validate --plan ... --acceptance ...`
and `expertflow run --plan ... --acceptance ...` commands. Search `execute`
requires a fresh execution output and the existing recommendation database.
Neither collection nor an accepted consumer adds authorization to an old study's
closed budget. Use each receipt's matching source/input snapshot; historical
receipts are not automatically portable to another checkout.

JSON decisions show covered identities, thread/graph defaults scope, retained
attempts and available collection/search phase costs. Automatic/manual search
cost remains 18 independent evaluations each, not a lower wall-time claim.
A valid reconstructed neutral/default-optimal outcome stays verified with
`utility_gain_established: false`, `selected_default: true` and its original
nonzero gate exit. Invalid inputs produce an identity/environment stop reason
and acquire no statistics or gain claim. Generate outputs are preparation,
not native performance evidence. Model digest caches last one synchronous
validation only; native run/execute behavior is unchanged.

[Wider coverage](superpowers/specs/2026-10-05-stock-coverage.md) registers Q4
and Granite with prose/code workloads: four cases, 107 calls each, 428 maximum.
They use the fixed existing controls/gates and report neutral/failed cases
individually. A separate reviewed collector and immutable source/live identity
freeze are required before running them. No wider utility result is claimed.

The compiler compares eligible configurations for one verified model, semantic
workload, runtime and host. A result is the strongest validated configuration in
its declared search space. A small sweep cannot establish a global optimum or
predict performance on a different host or workload.

## What exists today

| Boundary | Implementation and evidence |
| --- | --- |
| Model accounting | ModelDescriptor, ModelIR and AdapterRegistry; verified Gemma4 Q6/Q4 and complete real GraniteMoE Q6 inventory |
| Runtime identity | RuntimeBinding verifies pristine manifest, binary/dependency/CUDA hashes and build flags |
| Host identity | GPU/driver plus CPU, RAM, OS, affinity, thread environment and complete power policy |
| Numerical eligibility | Separate audited Gemma Q6/Q4 and Granite Q6 providers with declared control and baseline scopes |
| Scheduling coverage | Physical/midpoint/logical/incumbent thread anchors, CUDA graphs on/off, explicit exclusions and untested counts |
| Valid stock incumbent | Own reference/product receipts; Gemma Q6/Q4 and Granite Q6 separately passed fresh paired acceptance |
| Search | Three seeded complete blocks, then independent ten-pair confirmation of one finalist |
| Recommendation | Atomic execution-plan/search-receipt publication and artifact-backed read-only validation |
| Live reuse coverage | Gemma Q6 search32, Q4 reference10/product20/search18, Granite reference10/product20/search38 independently audited on one host/build |

Q6 scheduling search completed32 runs and retained the accepted 12-thread/graphs-on
incumbent. The16-thread screening finalist was0.985% slower in independent
confirmation, CI95 [-1.383%,-0.551%]. See
[the audited evidence report](evidence/stock-discovery-20261004/report.md).
Q4 is another quantization of Gemma. Subsequent real GraniteMoE Q6 validation
established second-family stock coverage with its own inventory, source proof,
GPU-resident reference and acceptance; see [Granite report](evidence/compiler-granite-20261004/report.md).
Synthetic topology/family fixtures still establish contracts rather than new
live hardware or placement coverage.

Q4's independent search completed 18 screening processes and retained the
accepted 12-thread/graphs-on incumbent, which ranked first. Its product validation
measured direct36.6095/sealed36.6385TPS with equivalence acceptance; screening
mean36.6867TPS is descriptive. No self-confirmation was required. See the
[Q4 evidence report](evidence/stock-discovery-20261004/q4-report.md).

## Reproducible sequence

1. Inspect metadata without loading full weights. Verify the actual file digest
   and normalize a complete tensor inventory through the family adapter.
2. Resolve pristine runtime and actual hardware. Capture the complete host;
   freeze model/workload/runtime/host identities before retaining measurements.
3. Obtain a verified stable reference and fresh paired stock-product acceptance.
   No historical TPS summary or claimed PASS boolean can satisfy this gate.
4. Admit settings only through reviewed operation-path evidence. Freeze exact
   candidates, exclusions, schedules, gates, process budget and source bundle.
5. Collect three complete blocks. Rank geometric candidate/incumbent ratios
   within each block; preserve descriptive variation and all raw evidence.
6. Confirm only the top challenger in twenty independent, balanced fresh runs.
   Require gain>=2%, CI95 lower>0, both CVs<=10%, exact tokens, owned memory,
   reserve and cleanup. If the gate fails, retain the accepted incumbent. If
   incumbent ranks first, consume no self-confirmation runs.
7. Independently reconstruct artifacts, ranking, statistics, receipt and plan.
   Publish coverage and exclusions alongside the configuration. Stop on partial
   collection, identity/correctness/environment failure; do not retry or discard.

The earlier incremental Q6 search coverage is threads 12/16 and graphs on/off, twelve screening
processes plus optional twenty confirmation processes. Threads8 is excluded by
the registered earlier rejected hypothesis; its old TPS is not reused as a
current-host sample. Other thread counts are untested. This exclusion belongs
to this experiment and is not a rule for other models or hardware.

## Earlier incremental Q6 search commands

Run from the matching frozen source checkout. Each collection needs a fresh
database and output directory. `generate` and `validate` launch no model process.
`run` starts the declared native experiment. The reviewed source revision is
f22bb54; do not edit measured source or protocol during collection.

```powershell
uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_search.py `
  --action generate `
  --output-dir C:/models/expertflow/runs/NEW-Q6-SEARCH/search `
  --manifest-output C:/models/expertflow/runs/NEW-Q6-SEARCH-preview.json

uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_search.py `
  --action run `
  --evidence-db C:/models/expertflow/runs/NEW-Q6-SEARCH/compiler.sqlite3 `
  --output-dir C:/models/expertflow/runs/NEW-Q6-SEARCH/search

uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_search.py `
  --action validate `
  --evidence-db C:/models/expertflow/runs/compiler-stock-search-20261004/compiler.sqlite3 `
  --recommendation C:/models/expertflow/runs/compiler-stock-search-20261004/search/recommended
```

These are usage examples, not authorization to repeat the registered experiment.
The actual search completed in `compiler-stock-search-20261004`.
Its preview is a zero-native preparation artifact. `run` creates a new manifest
with a fresh experiment identity, time and root; it does not execute preview
samples or reuse a preview as retained evidence.

The collector accepts explicit `--descriptor`, `--inventory`, `--hardware`,
`--workload`, `--runtime-identity`, `--source-plan`, `--source-receipt`,
`--source-evidence-db` and `--source-repository` inputs. Those arguments do not
bypass unsupported family/quantization, numerical provider or acceptance checks.

`--action execute` consumes a published recommendation and its existing evidence
database, with a fresh `--output-dir` outside the original experiment. It launches
the recommendation's tested thread/graph settings, verifies fresh token/memory/
cleanup/ownership/host/source evidence, and writes `accepted-execution.json`.
Its single TPS does not alter acceptance or the registered search budget. Actual
model, semantic workload and runtime inputs must match. Input tuning knobs do
not override the recommendation's tested settings.

For another supported topology, `--space-config` names a JSON object with exactly
`policy`, `excluded_threads` and `maximum_native_processes`. An explicit six-
candidate space has this shape:

```json
{"policy":"explicit","excluded_threads":[],"maximum_native_processes":38}
```

Candidate coverage is derived from topology anchors; budget must equal three
times the number of candidates plus twenty. A different host requires its own
actual hardware/host capture and fresh accepted incumbent. Configuration JSON
does not establish eligibility or allow use of this host's receipt elsewhere.

## Extending model support

Implement `ModelAdapter.normalize(descriptor, inventory, identity)` to produce
complete routed layer/component accounting. Register reviewed code through
`AdapterRegistry.register`; unknown families fail before measurement. Test
malformed provenance, topology/expert accounting and model identity mismatches.
Trusted Python callers can pass `adapter_registry=registry` to `inspect_model`
or `load_compiler_inputs`. CLI descriptors do not execute arbitrary plugin code;
its normal path resolves reviewed builtin adapters/providers. A registered
adapter does not establish numerical eligibility or supply missing weights.

Separately implement `SchedulingEligibilityProvider.attest(inputs, host,
repository)` and register through `EligibilityRegistry.register`. Bind verified
model bytes, runtime binaries/build flags, host architecture and immutable
operation-source objects. Explain which controls preserve complete-output
arithmetic, including any repacking, reductions and prompt path. Trusted
registration is a code boundary; arbitrary manifest provider IDs are rejected.
The native guard still requires exact prompt/generated tokens against the
model's own reference. A source proof plus one measured prompt does not certify
an unrelated quantization, backend or placement.

Freeze the provider and new driver source along with the experiment. Keep source
checkouts used by existing receipts available for historical reconstruction;
changed validators must not silently reinterpret frozen native evidence.

A clean detached checkout of the measured Q6 revision is preserved at
`C:/models/expertflow/worktrees/compiler-stock-search-f22bb54`. After collection
is terminal, historical validation can use the existing Python environment
while selecting that checkout's source explicitly:

```powershell
Set-Location C:/models/expertflow/worktrees/compiler-stock-search-f22bb54
$env:PYTHONPATH = 'C:/models/expertflow/worktrees/compiler-stock-search-f22bb54/src'
& C:/sem4/expertflow/.venv/Scripts/python.exe scripts/benchmark_compiler_stock_search.py `
  --action validate `
  --evidence-db C:/models/expertflow/runs/compiler-stock-search-20261004/compiler.sqlite3 `
  --recommendation C:/models/expertflow/runs/compiler-stock-search-20261004/search/recommended
```

The command's model/runtime/host checks still apply. A historical receipt does
not authorize reuse after actual model, runtime or host changes. Keep the
frozen checkout clean; ongoing implementation remains on ef-v2 in the main
checkout. Its actual validation result is recorded in the evidence ledger.

The [Q4 reuse plan](superpowers/plans/2026-10-04-stock-method-reuse.md) adds a fresh
ten-run reference, twenty-run product validation and six-configuration search
with at most38 processes. Its total maximum is68, conditional on successive
gates. The reference collector, CLI and Q4 source provider are implemented with
targeted contract checks; the actual zero-native preview verified complete
source/host/runtime/model binding. Independent review identified and resolved
a direct API runtime identity gap; post-fix full checks passed670 tests, with
seven source-environment skips and six pinned source checks passing separately.
Q4 reference/product/search native validation completed and passed independent
audit and actual CLI validation. This is separate Q4 evidence, not a Q6
quality-preserving speedup or a second-family result.

## Coverage and invalidation

### Q4 collection commands

Run from the ef-v2 checkout after reviewed source checks pass. These commands
create fresh experiments; existing output/database paths stop execution rather
than resume or overwrite evidence. Run each stage only after the preceding
stage's independent audit and CLI validation succeed. Q4 is a distinct model
artifact and provides no claim of preserving Q6 quality.

```powershell
$q4Inputs = @(
  '--descriptor', 'configs/compiler/gemma4-q4-model.json',
  '--inventory', 'docs/evidence/stock-discovery-20261004/q4-tensor-inventory.json',
  '--hardware', 'docs/evidence/compiler-phase3/inputs/hardware.json',
  '--workload', 'configs/compiler/gemma4-q6-single-request.json',
  '--runtime-identity', 'docs/evidence/stock-discovery-20261004/q4-runtime-identity.json'
)
$q4Source = 'C:/models/expertflow/worktrees/llama-q6-placement-final'
$q4Reference = 'C:/models/expertflow/runs/compiler-q4-stock-reference-20261004'
$q4Product = 'C:/models/expertflow/runs/compiler-q4-stock-product-20261004'
$q4Search = 'C:/models/expertflow/runs/compiler-q4-stock-search-20261004'

# Metadata normalization only; this does not prove actual weight hashes:
uv run --extra dev --extra quality --extra predictor expertflow inspect `
  --descriptor configs/compiler/gemma4-q4-model.json `
  --inventory docs/evidence/stock-discovery-20261004/q4-tensor-inventory.json `
  --output C:/models/expertflow/runs/NEW-Q4-ModelIR.json

# Optional read-only preparation before the fresh native run:
uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_reference.py @q4Inputs `
  --action generate --source-repository $q4Source `
  --output-dir "$q4Reference/reference" --manifest-output C:/models/expertflow/runs/NEW-Q4-reference-preview.json

uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_reference.py @q4Inputs `
  --action run --source-repository $q4Source `
  --evidence-db "$q4Reference/compiler.sqlite3" --output-dir "$q4Reference/reference"
uv run --extra dev --extra quality --extra predictor python docs/evidence/stock-discovery-20261004/verify_reference.py `
  --root "$q4Reference/reference" --database "$q4Reference/compiler.sqlite3" `
  --output docs/evidence/stock-discovery-20261004/q4-reference-verification.json
uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_reference.py @q4Inputs `
  --action validate --source-repository $q4Source `
  --evidence-db "$q4Reference/compiler.sqlite3" --reference-dir "$q4Reference/reference/diagnostic"

# Only after REFERENCE-STABLE and both reference checks pass:
uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_refinement.py @q4Inputs `
  --experiment stock-product --source-plan "$q4Reference/reference/diagnostic/execution-plan.json" `
  --source-evidence-db "$q4Reference/compiler.sqlite3" `
  --evidence-db "$q4Product/compiler.sqlite3" --output-dir "$q4Product/product"
uv run --extra dev --extra quality --extra predictor python docs/evidence/stock-discovery-20261004/verify_product.py `
  --root "$q4Product/product" --database "$q4Product/compiler.sqlite3" `
  --source-plan "$q4Reference/reference/diagnostic/execution-plan.json" `
  --output docs/evidence/stock-discovery-20261004/q4-product-verification.json
uv run --extra dev --extra quality --extra predictor expertflow validate @q4Inputs `
  --evidence-db "$q4Product/compiler.sqlite3" `
  --plan "$q4Product/product/accepted/execution-plan.json" `
  --acceptance "$q4Product/product/accepted/acceptance-receipt.json"

# Only after PASS-STOCK-FALLBACK and independent product/receipt checks pass:
uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_search.py @q4Inputs `
  --action run --source-repository $q4Source `
  --source-plan "$q4Product/product/accepted/execution-plan.json" `
  --source-receipt "$q4Product/product/accepted/acceptance-receipt.json" `
  --source-evidence-db "$q4Product/compiler.sqlite3" `
  --space-config configs/compiler/gemma4-q4-stock-search-space.json `
  --evidence-db "$q4Search/compiler.sqlite3" --output-dir "$q4Search/search"
uv run --extra dev --extra quality --extra predictor python docs/evidence/stock-discovery-20261004/verify_search.py `
  --root "$q4Search/search" --database "$q4Search/compiler.sqlite3" `
  --output docs/evidence/stock-discovery-20261004/q4-search-verification.json
uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_search.py @q4Inputs `
  --action validate --evidence-db "$q4Search/compiler.sqlite3" --recommendation "$q4Search/search/recommended"
```

The explicit Q4 space includes threads8/12/16 and both graph modes on this
host. Q6's rejected eight-thread condition is not an exclusion for Q4. Stop
after any failed gate; preserve all failed artifacts without retries.

After recommendation validation, the search driver's `--action execute` is
available with the same explicit model inputs, existing search database and
recommendation, plus a fresh output directory outside the original experiment.
It is a consumer operation, not part of Q4's registered68-process validation
budget; no additional native execution is included in these collection commands.

| Change or setting | Required action |
| --- | --- |
| Model bytes, quantization or inventory | New normalization, scope proof and own reference/validation |
| Prompt, context, prediction length, concurrency or objective | New semantic experiment; no TPS comparison across mismatched objectives |
| Runtime, CPU SIMD, dependencies or CUDA runtime | Verify new binaries and operation path; invalidate old launch identity |
| GPU, driver, CPU topology, affinity, RAM, OS or power/environment | Capture new host and validate a fresh incumbent |
| Partial affinity, multiple sockets or >64 logical processors | Explicit unsupported capability; add reviewed topology/control support first |
| Expert placement/offload, batching or KV format changes | Separate numerical eligibility and quality protocol before admission |
| Other family/quantization | Adapter and reviewed provider, then separate live evidence |

Historical stock22.9667TPS and server confirmation24.411/replay25.383TPS are
descriptive comparisons across different experiments. The old replay failed
its original gate. Static28.13TPS failed its quality gate; aggregate four-slot
35.6699TPS measures another objective. None is a matched control or an eligible
speed target obtained by relaxing correctness. See the retained
[historical audit](evidence/stock-discovery-20261004/history-audit.json).

The [final objective audit](evidence/stock-discovery-20261004/objective-audit.md)
maps the registered StageA/B/C requirements to evidence.
[Final verification](evidence/stock-discovery-20261004/method-verification.json)
rechecked all100 distinct native records, artifacts and frozen source revisions.
This completes the registered method, not universal live support or an eligible
Q6 speedup. Subsequent families/settings require the extension procedure above.
