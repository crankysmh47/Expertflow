# Finding a validated stock MoE configuration

The compiler compares eligible configurations for one verified model, semantic
workload, runtime and host. A result is the strongest validated configuration in
its declared search space. A small sweep cannot establish a global optimum or
predict performance on a different host or workload.

## What exists today

| Boundary | Implementation and evidence |
| --- | --- |
| Model accounting | ModelDescriptor, ModelIR and AdapterRegistry; real Gemma4 Q6, Q4 header inspected separately |
| Runtime identity | RuntimeBinding verifies pristine manifest, binary/dependency/CUDA hashes and build flags |
| Host identity | GPU/driver plus CPU, RAM, OS, affinity, thread environment and complete power policy |
| Numerical eligibility | Provider registry; Q6 audited native search, Q4 separately audited source/verified zero-native preview with review/live validation pending |
| Scheduling coverage | Physical/midpoint/logical/incumbent thread anchors, CUDA graphs on/off, explicit exclusions and untested counts |
| Valid stock incumbent | Fresh paired product receipt; Q6 completed twenty runs and passed |
| Search | Three seeded complete blocks, then independent ten-pair confirmation of one finalist |
| Recommendation | Atomic execution-plan/search-receipt publication and artifact-backed read-only validation |
| Remaining reuse work | Q4 reference/product/search native validation, full checks/review and scope audit |

Q6 scheduling search completed32 runs and retained the accepted12-thread/graphs-on
incumbent. The16-thread screening finalist was0.985% slower in independent
confirmation, CI95[-1.383%,-0.551%]. See
[the audited evidence report](evidence/stock-discovery-20261004/report.md).
Q4 is another quantization of the same real family; synthetic second-family
fixtures establish plugin contracts only.

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

The current Q6 coverage is threads12/16 and graphs on/off, twelve screening
processes plus optional twenty confirmation processes. Threads8 is excluded by
the registered earlier rejected hypothesis; its old TPS is not reused as a
current-host sample. Other thread counts are untested. This exclusion belongs
to this experiment and is not a rule for other models or hardware.

## Current Q6 commands

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
Q4 native validation remains pending. No Q4 speed or accepted product is claimed yet.

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

# Only after PASS-STOCK-FALLBACK and independent product/receipt checks pass:
uv run --extra dev --extra quality --extra predictor python scripts/benchmark_compiler_stock_search.py @q4Inputs `
  --action run --source-repository $q4Source `
  --source-plan "$q4Product/product/accepted/execution-plan.json" `
  --source-receipt "$q4Product/product/accepted/acceptance-receipt.json" `
  --source-evidence-db "$q4Product/compiler.sqlite3" `
  --space-config configs/compiler/gemma4-q4-stock-search-space.json `
  --evidence-db "$q4Search/compiler.sqlite3" --output-dir "$q4Search/search"
```

The explicit Q4 space includes threads8/12/16 and both graph modes on this
host. Q6's rejected eight-thread condition is not an exclusion for Q4. Stop
after any failed gate; preserve all failed artifacts without retries.

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
