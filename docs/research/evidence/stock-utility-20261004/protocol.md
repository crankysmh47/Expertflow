# Stock utility study

Authorized execution follows the scoped
[placement feasibility no-go](../placement-proof-20261004/feasibility.md).
The authoritative [frozen specification](../../superpowers/specs/2026-10-04-stock-utility-proof.md)
declares all gates, controls, inputs, ranks and budgets. Implementation is
`scripts/benchmark_compiler_stock_utility.py`. Main collection completed from
measured source checkpoint `0370579`; [frozen manifest](main-frozen-manifest.json)
pins the actual source, inputs, native eligibility, host and process budget.
Terminal result: [PRODUCT-VALIDATION-STOP](report.md), 106 retained processes.
The utility comparison passed; fresh product equivalence was inconclusive.
Consumer and held-out stages did not run. This study is closed.

## Inputs and gates

Pinned Gemma Q6 weights, pristine a7312ae runtime, CUDA12.8 and the current
Windows RTX5060Ti16GB host. Main input is
`configs/compiler/gemma4-q6-single-request.json`; transfer is
`configs/compiler/gemma4-q6-utility-transfer.json`. Both prompts were committed
before collection; exact native generated tokens must match the own reference
for each workload. Reviewed controls are threads8/12/16 and CUDA graphs on/off.

Defaults mean resolved thread/graph defaults (8/on), with CPU-MoE fit and
context4096/predict512/seed42/F16/workload controls held fixed. The manual control
is an independent scripted six-configuration grid. This is not a comparison
with every out-of-box stock setting or every human tuning strategy.

At least5% paired geometric TPS gain over defaults, CI95 lower>0, both
comparisons' arm CVs<=10%, manual equivalence CI90 strictly within±2%, and
automatic evaluation count no greater than manual are required. Both searches
spend18 native evaluations; native phase time is separately reported.

Per workload: reference10, automatic18, manual18, default/selected pairs20,
manual/selected pairs20, then conditional paired product20 and consumer1.
Maximum107; utility gate consumes86. The second workload is conditional on
main utility/product/consumer acceptance, for a maximum214 across both.
No retries, ordinary-sample discards, extra finalists or source/gate changes.

Original utility source/host/manifest guards cover all launches, including
product and consumer. An attempt is persisted before calling the runner, with
observed native process identity retained on failure. Confirmed process starts
and attempted launches are counted separately. All successful final results
must reconstruct from native artifacts and existing acceptance receipts.

## Collection and validation

The recorded collection command is below; do not rerun the closed study.
Future studies require their own justified protocol and fresh paths.

```powershell
uv run --no-sync python scripts/benchmark_compiler_stock_utility.py --action run --descriptor configs/compiler/gemma4-q6-model.json --inventory docs/evidence/q6-download/tensor-inventory.json --hardware docs/evidence/compiler-phase3/inputs/hardware.json --workload configs/compiler/gemma4-q6-single-request.json --runtime-identity docs/evidence/compiler-phase3/inputs/runtime-identity.json --source-repository C:/models/expertflow/worktrees/llama-q6-placement-final --evidence-db C:/models/expertflow/runs/compiler-stock-utility-main-20261004/utility.sqlite3 --output-dir C:/models/expertflow/runs/compiler-stock-utility-main-20261004/utility
```

Reconstruct with the same arguments and `--action validate`. It checks supplied
inputs as well as evidence, returns `PRODUCT-VALIDATION-STOP` and exits nonzero
for this negative terminal result. See [actual output](main-validation.log).
The independent raw audit launches no model process:

```powershell
uv run --no-sync python docs/evidence/stock-utility-20261004/independent_audit.py --report C:/models/expertflow/runs/compiler-stock-utility-main-20261004/utility/report.json --database C:/models/expertflow/runs/compiler-stock-utility-main-20261004/utility.sqlite3 --output C:/models/expertflow/runs/compiler-stock-utility-main-20261004/reconstructed-audit.json
```

It reconstructs the utility and product statistics and verifies all 106 retained
native records. Native logs, artifacts and SQLite databases remain under the
run root; [published audit](main-audit.json) pins their report/database signatures.

The registered conditional transfer, which was not activated, would use the
transfer workload and fresh root
`C:/models/expertflow/runs/compiler-stock-utility-transfer-20261004` without
changing any collector, compiler, prompt, search-space or acceptance source.
Public CLI consolidation remains deferred until complete utility/product proof.
