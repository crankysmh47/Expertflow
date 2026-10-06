# Wider stock control eligibility: scoped no-go

Decision, 2026-10-06: **NO-GO for admitting a new useful exact control from
expert/layer offload, flash attention, or batch/microbatch sizing under the
current workload and provider contracts.** This read-only audit made **zero
native calls**. It establishes neither universal impossibility nor measured
quality loss or performance for these controls.

The completed [wider utility study](../stock-coverage-20261005/report.md) remains
closed at four `NO-UTILITY-GAIN` outcomes. The earlier qualified Q6 stock result
retains its narrow scope. Neither verdict admits more controls or transfers
unused experiment budgets.

## Bound source and workload

The audit used exact Git objects from pristine upstream
`a7312ae94f801fc9c6786dc56e38df57b964f697`, rather than the registered source
repository's current fork checkout. Sixteen complete source objects are bound
by Git blob ID and raw SHA-256 in [the initial inventory](source-inspection.json)
and [supplemental inventory](supplemental-source-objects.json), with complete
[initial](upstream-source-objects.zip) and
[supplemental](supplemental-source-objects.zip) archives. The
[17 operation excerpts](source-excerpts.md) retain original line numbers.
Current provider and plan contracts are hashed separately in the inventory.

The four native reference prompts contain 39, 45, 41 and 46 tokens respectively;
tokenization counts agree with native completion counters. Each workload has
concurrency one, 512 generated tokens, context 4,096, batch cap 2,048 and
microbatch cap 512. The objective is serial `decode_tps`, policy `exact`, with
no approximate quality budget. Existing admitted controls are exactly
`threads` and `cuda_graphs` for each of the three trusted providers.

## Control decisions

- **Expert/layer offload changes numerical paths.** `llama-model.cpp:1293-1304`
  selects CPU/GPU layer buffers; `common/arg.cpp:2472-2495` and
  `llama-model-loader.cpp:1162-1188` implement expert CPU buffer overrides.
  CPU Q4/Q6 dot products use Q8_0/Q8_K activations, while the inspected CUDA
  MMVQ path quantizes activations to Q8_1. CUDA dispatch also depends on routed
  tensor shape. These are distinct backend/arithmetic paths, without a new
  complete prompt/decode equivalence proof. No control is admitted, and the
  historical placement quality stop is not reopened.
- **Flash attention changes the operator/reduction path.**
  `llama-graph.cpp:2405-2429` selects fused attention, with conditional K/V casts;
  `2449-2506` builds separate matmul/softmax/value operations. Mask storage and
  fused CUDA tile selection differ too. An F32 precision setting alone does
  not prove complete arithmetic equivalence. No new numerical error or quality
  loss was measured, and this audit does not resolve every baseline's effective
  flash-attention setting. Admission needs a new path qualification.
- **Nonbinding batch caps have no demonstrated decode benefit here.**
  `llama-context.cpp:238-246,1750-1760` caps and passes microbatch sizes;
  `llama-batch.cpp:476-508` splits batches. The nonspeculative server path adds
  one sampled token per active slot (`server-context.cpp:448-457`). These short
  prompts fit the existing cap, and serial decode processes one token per step.
  Merely changing a nonbinding cap does not reduce that work; allocation and
  startup effects are unmeasured. Active prompt fragmentation can change tensor
  shapes and CUDA dispatch and needs its own numerical proof. Longer prompts,
  concurrency, serving, or a prefill/latency objective need a new registration.

## Verification and next gate

The existing provider/plan guard suite passed **55 tests** in 0.19 seconds;
[the log](contract-checks.log) has line endings normalized to LF. These are CPU
contract checks, not new native correctness, quality or speed evidence.

[Read-only verification](verification.json) passed: all 16 archived Git objects,
63 unchanged wider source/input files, 344 retained wider native starts, and
the original 41 frozen files, six history pins and 148 native starts. Both
report digests are unchanged. Reproduce from the repository root with the
registered local source repository and native artifacts available:

```powershell
uv run --no-sync python docs/evidence/stock-control-scope-20261006/verify_audit.py
```

**Further native optimization is blocked by missing numerical qualification
and a demonstrated useful mechanism for an additional control.** An exact
extension needs reviewed complete prompt/decode arithmetic equivalence plus
a concrete benefit mechanism, followed by a separate bounded registration.
Approximation instead needs a separate quality policy/provider, datasets,
held-out gates, matched controls, fixed budget and full product/consumer
acceptance. Neither prerequisite exists for these new controls today.
Do not retune the closed thread/graph grids, relax the 5% gate, or reuse the
wider study's 84 unspent conditional calls. This closes the authorized audit
with a scoped no-go and records the prerequisite for further native work.
