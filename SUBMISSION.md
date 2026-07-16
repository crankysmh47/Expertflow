# ExpertFlow Build Week submission

## One-line pitch

ExpertFlow tells you whether a sparse MoE model's routed experts are worth
caching on your actual GPU before you spend days changing the runtime.

## What judges should run

The fastest path is CPU-only and does not require model weights:

```powershell
uv sync --extra dev
powershell -ExecutionPolicy Bypass -File submission\verify.ps1
```

Then open:

```text
submission\observatory.html
```

The report is self-contained. It shows the measured hardware envelope,
model-specific expert locality, cache-policy estimates, transfer evidence,
held-out prompt/domain results, and the machine recommendation.

## Why this is the submitted product

The project plan required the Observatory path whenever the live predictive
runtime failed to beat the relevant baseline. That is what happened.

The runtime work was still useful. We built exact packed-Q4 expert slots,
replacement, a 32-slot LRU, CUDA-resident multi-layer caching, a frozen temporal
predictor, and a two-slot asynchronous sidecar. The final projected-state filter
even prevented real blocking misses. But across the fixed general, code, and
translation matrix, decode TPS was 1.15% lower and end-to-end time was 0.43%
slower than the matched reactive runtime.

So the release makes one clean claim: ExpertFlow is a working observatory and
decision tool. It does not claim a predictive runtime speedup.

## Main evidence

| Result | Evidence |
|---|---|
| Real model | Gemma 4 26B A4B Q4_0, pinned 14,439,361,440-byte artifact |
| Real machine | RTX 5060 Ti, 16,311 MiB reported VRAM |
| Exact expert layout | 3,840 layer-expert objects measured; aligned slot 3,346,048 bytes |
| Held-out locality | Static-96 87.57%, conversation-reset LRU 86.34% |
| Split discipline | Static set selected from training conversations only; validation/test untouched |
| CUDA transfer | 0.234016 ms p50 / 0.234272 ms p95 for an aligned pinned expert copy |
| Projected cache | 6,433.14 MiB across 21 target layers, with 800.86 MiB configurable reserve in the frozen profile |
| Judge replay | 8 events, 64 demands, 26 static hits, 19 LRU hits |
| Runtime correctness | Exact cache-off/cache-on tokens and router selections in accepted milestones |
| Release default | `live_cache_enabled=false` |

Measured, simulated, projected, oracle, standalone CUDA, Vulkan, and live-runtime
numbers are labeled separately throughout the evidence.

## Three-minute demo

Use [submission/demo-script.md](submission/demo-script.md). It follows the
Observatory-only script selected by the project gate:

1. show the expensive systems mistake;
2. inspect the machine and model;
3. explain real routing locality;
4. replay cache policies on a causal timeline;
5. show the machine recommendation;
6. close with the honest runtime decision.

## Repository map

- [Bundled Observatory](submission/observatory.html)
- [Judge instructions](submission/README.md)
- [Final scorecard](submission/final-scorecard.md)
- [Demo script](submission/demo-script.md)
- [Runtime freeze](docs/evidence/live-cache/runtime-freeze.md)
- [Final predictive result](docs/evidence/live-cache/t2-projected-state-policy-result.md)
- [Append-only project log](PROJECT_LOG.md)

The protected Observatory floor remains at `d846bdf`. Release integration is
kept on a separate branch.
