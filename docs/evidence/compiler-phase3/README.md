# Phase 3 compiler evidence

Reproduce the exact single-request compiler gate on `ef-v2`.

The live verdict is **VALIDATION-STOP**. The selected stock settings were
`-ngl 99 --cpu-moe`. Ten confirmation runs averaged **24.411 TPS** with
**3.901% CV**. Pending-plan replay produced **25.383 TPS**, a **3.980%** increase,
outside the frozen absolute **2%** repeatability tolerance. Token, memory,
ownership and cleanup gates passed, but no validated plan was published.

This faster replay is a repeatability failure, not a slowdown. The fixed-budget
experiment stops here; no reruns were made to seek a passing sample.
`verification.json` records the verdict and `measurement-manifest.json` binds
all 27 independent runs to their raw artifact hashes. The pending plan is
retained only as a diagnostic artifact.

## Environment

Run from the repository root in PowerShell:

```powershell
uv sync --frozen --extra dev --extra quality --extra predictor
uv run --extra dev --extra quality --extra predictor pytest -q
uv run --extra dev --extra quality --extra predictor python -m compileall -q src/expertflow
git diff --check
```

The current CPU gate passed 481 tests. Six historical source-contract modules
skip without an external source checkout. When `source_path` is supplied, live
compilation checks its fork revision and runs the two applicable static/profile
suites before starting GPU inference. Temporal-cache source contracts belong to
a different fork.

External files are pinned by the model manifest, runtime manifests, and
`preflight.json`:

- Q6 GGUF: `C:/models/gemma-4-26b-a4b-q6/google_gemma-4-26B-A4B-it-Q6_K.gguf`.
  Size 22,862,575,520 bytes; SHA-256
  `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.
- Pristine binaries: `C:/models/expertflow/builds/llama-a7312ae-cuda128-clean/bin`.
- Fork binaries: `C:/models/expertflow/builds/llama-q6-placement-final/bin`.
- Fork source: `C:/models/expertflow/worktrees/llama-q6-placement-final`.
- CUDA runtime: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/cudart64_12.dll`.

All companion DLLs and six patches have expected hashes. Changing a binary,
implementation DLL, CUDA runtime, model or supplied source revision fails the
identity gate.

## Inputs and recorded reproduction

`inputs/runtime-identity.json` names both manifests, binary directories, CUDA
runtime and model identity. `inputs/hardware.json` freezes the GPU identity,
driver and allocation frontier captured for this run. For a new machine or
driver, capture a fresh hardware snapshot rather than editing measured evidence.
The compiler rechecks the physical GPU, driver and current free reserve live.

The descriptor is `configs/compiler/gemma4-q6-model.json`. The inventory is
metadata-derived and covers all 30 routed layers. Inspection reads metadata;
compilation verifies the actual model's full size and SHA-256.

```powershell
uv run --extra dev --extra quality --extra predictor expertflow inspect --descriptor configs/compiler/gemma4-q6-model.json --inventory docs/evidence/q6-download/tensor-inventory.json --output docs/evidence/compiler-phase3/model-ir.json
```

The following PowerShell arguments are shared by compile, validate and replay:

```powershell
$compilerInputs = @(
  '--descriptor', 'configs/compiler/gemma4-q6-model.json',
  '--inventory', 'docs/evidence/q6-download/tensor-inventory.json',
  '--hardware', 'docs/evidence/compiler-phase3/inputs/hardware.json',
  '--workload', 'configs/compiler/gemma4-q6-single-request.json',
  '--runtime-identity', 'docs/evidence/compiler-phase3/inputs/runtime-identity.json',
  '--evidence-db', 'C:/models/expertflow/runs/compiler-phase3-20261003-reviewed/compiler.sqlite3'
)
$compilerProfiles = @(
  '--layer-profile', 'C:/models/expertflow/runs/q6-placement-final/stage1-profiles/run-1/split-profile.json',
  '--layer-profile', 'C:/models/expertflow/runs/q6-placement-final/stage1-profiles/run-2/split-profile.json',
  '--layer-profile', 'C:/models/expertflow/runs/q6-placement-final/stage1-profiles/run-3/split-profile.json'
)
uv run --extra dev --extra quality --extra predictor expertflow compile @compilerInputs @compilerProfiles --output-dir docs/evidence/compiler-phase3/recorded-new --recorded-evidence docs/evidence/q6-placement-final/results.json
```

This historical replay returned `RECORDED-DIAGNOSTIC`, with no sealed plan.
It preserves the earlier strongest stock result of 22.966667 TPS separately from
the later matched OFF/ON means of 22.28/28.13 TPS. The historical CLI workload
used context 2048 and `Caching.`; the product workload uses context 4096 and the
exact bytes in `configs/baseline-prompt.txt`. Historical response hashes differ,
native token IDs are absent, and the strict quality gate failed. Those records
cannot establish exact static execution or a product server target.

## Live compilation and replay

Use a fresh output directory for each invocation:

```powershell
uv run --extra dev --extra quality --extra predictor expertflow compile @compilerInputs @compilerProfiles --output-dir C:/models/expertflow/runs/compiler-phase3-20261003-reviewed/live-new
```

Compilation checks Compute/CUDA engine activity before each owned process. The
WDDM compute-process listing also includes GUI applications; an entry alone is
not evidence of active compute. Failed counters and unknown memory do not count
as zero usage. The runner samples owned-PID dedicated memory every 200 ms and
requires device-free reserve and settled teardown. Its raw process manifest
separates native child exit from successful completion followed by owned
termination, because Windows termination commonly returns child exit code 1.

The stock matrix includes `auto`, `all` and `99`, each with and without CPU-MoE.
Resolved duplicates retain warmup diagnostics but share one measured
representative. Initial candidates get one discarded warmup and three retained
cold-process measurements. An unsafe allocation is rejected. Missing required
instrumentation blocks the run. Ordinary slow measurements remain in evidence.

The pinned static implementation changes the CPU-to-CUDA numerical path. Static
placements are reported with aligned bytes, greedy ranks and runtime caps, then
rejected with `numerical_path_change`. A finite token match cannot override this
exactness restriction. No approximate profile is enabled by this plan.

The selected stock candidate gets ten retained confirmation runs. Only after
token stability, memory, cleanup, variance and evidence checks pass does the
compiler write a pending sealed plan and reload it for replay. Replay must be
within 2% of the confirmation mean. `execution-plan.json` is published only
after that replay passes; a pending plan is not a live-validated product.

For a future published plan (these commands were not run for this failed gate):

```powershell
uv run --extra dev --extra quality --extra predictor expertflow validate @compilerInputs --plan C:/models/expertflow/runs/compiler-phase3-20261003-reviewed/live/execution-plan.json
uv run --extra dev --extra quality --extra predictor expertflow run @compilerInputs --plan C:/models/expertflow/runs/compiler-phase3-20261003-reviewed/live/execution-plan.json --output-dir C:/models/expertflow/runs/compiler-phase3-20261003-reviewed/replay-new
```

`run --plan` performs no search. The existing positional `run DEPLOYMENT`
interface remains available; mixing the two interfaces is rejected.

## Evidence and verdicts

The append-only SQLite store and raw process logs are under
`C:/models/expertflow/runs/compiler-phase3-20261003-reviewed/`. The earlier
`compiler-phase3-20261003` run was stopped for review fixes and remains diagnostic.
Raw records bind actual argv, effective runtime controls and the ModelIR snapshot
to frozen identities. They contain the exact tokenization input, native token
arrays, requests, completions, native timings, separate whole-request wall time,
owned memory observations and valid absent-memory teardown. OS process creation
time plus PID identifies each independent run; the store rejects duplicate runs.
Older databases without this proof require a fresh database.
All JSON uses UTF-8 and LF so checkout normalization cannot change its hashes.
Summary artifacts here reference the raw paths and hashes; external weights,
binaries and the local measurement database are not distributed in Git.

| Verdict | Meaning |
| --- | --- |
| `PASS-STOCK-FALLBACK` | Exact measured stock passes confirmation and sealed replay; static no-go remains explicit. |
| `PASS-STATIC` | Reserved for a future eligible exact static path with a positive paired lower confidence bound. This pinned runtime cannot produce it. |
| `VALIDATION-STOP` | Selected-plan identity, token, memory, cleanup or replay validation failed. |
| `INCONCLUSIVE` | Retained variance or uncertainty prevents a valid floor. |
| `ENVIRONMENT-BLOCKED` | A required artifact, idle GPU or ownership/instrumentation capability is unavailable. |

No verdict authorizes later CUDA, KV, adapter, speculative or dynamic research
tracks. See the repaired plan and compiler design for their separate scope.

## Further reading

- [Implementation plan](../../superpowers/plans/2026-08-28-inference-compiler-spine.md)
- [Compiler design](../../superpowers/specs/2026-08-28-expertflow-inference-compiler-design.md)
- [Historical placement evidence](../q6-placement-final/README.md)
- [Repository license](../../../LICENSE)
