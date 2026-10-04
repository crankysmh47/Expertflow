# Real second-family stock compiler result

Granite generalization passed its own reference and paired-product gates.
The bounded scheduling search retained **12 threads, CUDA graphs on**, with
**ngl99/CPU-MoE false**, exact F16 KV, batch2048/ub512 and the frozen single-request
512-token workload. This is an audited second architecture on the pinned host and
runtime. It establishes reproducible stock-plan acceptance, without a confirmed
thread optimization or a universal model/hardware claim.

Measured source: `7cbae5dbb0875927e7a620cf3784fcb1bb930823` on `ef-v2`.
Frozen checkout: `C:/models/expertflow/worktrees/compiler-granite-7cbae5d`.
All 68 allowed native processes were retained once: reference10, product20,
screening18, confirmation20. No retries, discarded samples, changed gates or
alternative finalists occurred. No model processes remained at completion.

## Artifact and implementation

Separately verified Granite3.1 1B-A400M Instruct Q6_K artifact:
1,099,212,096bytes, SHA
`4566cfa92be10888026bd3663c83d64e91cd91f874dfb3607596587ff1c8f67f`.
Publisher revision `940d2e1f9f65330615c7c8e980e6c5ac73d3360c`, with separately
pinned official IBM config, is recorded in [artifact-pin.json](artifact-pin.json).
The complete242-tensor inventory has24layers/32experts/top8 and72Q6_K routed
tensors. ModelIR SHA
`54f52d76dba171c90ee73150688351fd8a88aa96373af6b4c66b8e00aa1fbbce`;
inventory SHA
`3de0a37570c518db90d027d256d2f5785d300582f5d68c2c5cfb6acbc132e569`.

A separate adapter validates full tensor shapes/types/byte accounting, offsets,
totals, provenance and attention/scaling metadata. Shared routed-bank accounting
preserves the historical Gemma ModelIR hash. A separate trusted provider binds
the actual artifact, complete inventory, pristine runtime, host capability and
18immutable upstream source objects. The provider explicitly declares the GPU
baseline; this placement selection uses the footprint, not a measured CPU/GPU
placement comparison. Only threads and graph capture may vary in this search.

Final full suite: **745passed,7skipped** in534.03s. Skips are optional historical
source-contract modules requiring their checkout environment. Actual live input
preflight ran its pinned native-source checks; Granite eligibility independently
verified all18immutable source objects. The required independent review found
twoP2gaps, full inventory validation and product source/spec freeze; one fix pass
resolved both,65focused tests passed, and the reviewer confirmed resolution and
unchanged ModelIR. See [implementation-verification.json](implementation-verification.json)
and [scheduling-source-audit.md](scheduling-source-audit.md).

## Reference and product acceptance

Ten own-model reference processes passed exact prompt/generated tokens, owned
memory/reserve, cleanup and frozen source/host. Mean486.2893687285578TPS;
sampleCV2.813250734496988%, below10%. The independent audit and actual CLI
returned VERIFIED-STOCK-REFERENCE and VALIDATED-STOCK-REFERENCE. Reference
diagnostic plan SHA `ad820b31268bce882a8ecc54dee050ad8f882a753554570ab0906b9eebac8a48`.

Twenty fresh balanced product processes passed equivalence and noninferiority:
direct481.7654442685547TPS, sealed479.4134673191409TPS; paired change
−0.4865279576758059%, CI90[−1.183393040523067%,+0.25397616142937374%],
CI95[−1.2933731413351781%,+0.39820334452666145%]. CVs1.016202442546693% and
0.816326578489624%. The independent audit and actual CLI returned VERIFIED-PASS
and VALIDATED-STOCK-FALLBACK. Accepted plan SHA:
`449f1dce2e7045523c1b4a542d89581435413cb30c10d4017ce5dfe821d79a7f`.
Equivalence does not establish an optimization gain.

## Scheduling search

Three complete seeded screening blocks covered six configurations:

| Threads | CUDA graphs | Screening mean TPS | Geometric change vs incumbent |
| --- | --- | --- | --- |
| 16 | on | 482.803616 | +1.063% |
| 8 | on | 479.562483 | +0.385% |
| 12 | on | 477.718366 | 0.000% |
| 12 | off | 248.223148 | −48.039% |
| 16 | off | 244.797650 | −48.759% |
| 8 | off | 244.700563 | −48.778% |

These are descriptive three-block results, not confirmed speedups. Threads16/on
ranked first and consumed the single20-process confirmation against12/on.
Confirmation gave −0.2860544398941481%, CI95[−1.4730362574499345%,+0.903908866403005%],
below the required2% point gain and positive CI95lower bound. Means were
480.7174420243085/479.34743877177766TPS and CVs1.277983012863544/1.3665910884283645%.
The top challenger was not accepted; no other finalist was tried.

RECOMMENDED-INCUMBENT preserves the accepted plan hash449f1dce. Independent audit
returned VERIFIED-STOCK-SEARCH; actual CLI returned VALIDATED-STOCK-RECOMMENDATION.
All38search processes passed own-reference tokens, complete launch identity,
memory/reserve, cleanup and ownership. Cross-stage verification rechecked all68
unique native processes, raw artifact hashes, source Git blobs, host identity,
prerequisite hashes and unchanged historical Gemma plans/terminal evidence.
See [generalization-verification.json](generalization-verification.json).

## Consumption and limits

Repository copies preserve the exact published bytes in `reference-diagnostic/`,
`accepted/` and `recommended/`. Raw artifacts and databases remain at:

- `C:/models/expertflow/runs/compiler-granite-reference-20261004`
- `C:/models/expertflow/runs/compiler-granite-product-20261004`
- `C:/models/expertflow/runs/compiler-granite-search-20261004`

From the unchanged measured source, read-only validation uses:

```powershell
uv run --no-sync python scripts/benchmark_compiler_stock_search.py --action validate `
  --descriptor configs/compiler/granite-q6-model.json `
  --inventory docs/evidence/compiler-granite-20261004/tensor-inventory.json `
  --hardware docs/evidence/compiler-phase3/inputs/hardware.json `
  --workload configs/compiler/granite-q6-single-request.json `
  --runtime-identity docs/evidence/compiler-granite-20261004/runtime-identity.json `
  --recommendation docs/evidence/compiler-granite-20261004/recommended `
  --evidence-db C:/models/expertflow/runs/compiler-granite-search-20261004/compiler.sqlite3
```

The published-copy validation was also performed. Changing model/inventory,
runtime/binaries, host power/topology/affinity, semantic workload, source/spec,
provider proof or raw evidence invalidates reuse. Use the corresponding frozen
measured checkout to reconstruct historical Gemma/Q4/PDL receipts after compiler
source changes. Frozen checkouts require their own environment; do not accidentally
import a newer editable installation when using them.

The registered68-process budget includes no extra post-publication native
`--action execute` run. Fresh sealed product arms already executed this exact
candidate; the public execution path has fake/CPU contract coverage. No additional
native launch was used to supplement the fixed experiment.

Live scope now covers Gemma4(Q6_K and separateQ4_0) and GraniteMoE(Q6_K) stock
contracts on this host/build. No Granite static placement/profile/cache/fork
execution, shared-expert variants, other weights/quantizations/builds, KV changes,
batching/concurrency/offload comparisons, other backends or global optimum are
established. Counts1–7,9–11,13–15 and other controls remain untested. Granite TPS
is not a Gemma quality/speed comparison. Gemma's audited PDL NO-GO remains closed
and its accepted Q6/Q4 plans are unchanged.
