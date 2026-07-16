# Expanded Next-Layer Predictor

## Verdict

The bounded offline predictor experiment passed its leakage and reproducibility
gates. Validation selected `b2_source_normalized_separate` with candidate width
12 and `observed_support` admission. The frozen 12-conversation test split was
then evaluated exactly once.

This is measured offline routing-prediction evidence plus simulated 32-slot
shadow-cache accounting. It is not a live-runtime, transfer-overlap, throughput,
or speedup result.

## Frozen data contract

- Manifest:
  `C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json`
- Manifest SHA-256:
  `b59ab020843a98121fca1f60227d3bf7272fe2a0e8f4b10c53add7182f7437fa`
- Canonical runtime: `expertflow-canonical-observer-v1`
- Trace generation: `trace_v2_canonical_segmented`
- Conversations: 60 train / 12 validation / 12 test
- Domain balance: 10/2/2 in each of six domains
- Adjacent-layer samples: 138,736 train / 27,405 validation / 28,043 test
- Join: conversation, forward, token index, token ID, and consecutive MoE layer
- Prompt hashes and conversation IDs are unique.
- Fit materialized only train and validation traces. The guarded test command
  materialized test traces only after the validation selection lock existed.

The frozen corpus definition is 47,147 bytes with SHA-256
`970681c0126cc5400524e5b4328f0ecaf87c72d346a7fd99896a44224720dbab`.
It records exact, normalized, and task-family near-duplicate policies. The
initial selection lock accidentally serialized the manifest's absent
`deduplication` field as null; this does not change the enforced unique prompt
hash check or selected configuration, but the corpus file above is the
authoritative deduplication-policy provenance.

## Validation-only selection

The search was fixed before validation: four B2 configurations, widths 8/12/16,
and `all_ranked` or `observed_support` admission. B3 and B4 retained their pilot
architectures and hyperparameters.

| Model | Recall@8 | Recall@12 | Recall@16 | CPU p95 |
|---|---:|---:|---:|---:|
| B0 copy | 6.61% | 9.58% | 12.71% | 13.7 us |
| B1 frequency | 25.49% | 33.40% | 39.73% | 11.5 us |
| B2 raw, pooled | 45.70% | 57.36% | 65.83% | 66.9 us |
| B2 raw, phase-separated | 48.17% | 59.93% | 68.11% | 53.2 us |
| B2 normalized, pooled | 52.77% | 65.08% | 73.09% | 91.5 us |
| **B2 normalized, phase-separated** | **55.31%** | **67.51%** | **75.03%** | **78.1 us** |
| B3 fixed linear | 42.54% | 54.19% | 62.21% | 5.7 us |
| B4 fixed shared MLP | 42.04% | 53.50% | 61.11% | 19.2 us |

For the selected B2, validation net ready gain after eviction regret was 17,090
at width 8, 19,012 at width 12, and 15,001 at width 16. This froze width 12.
Neither learned model met the fixed two-percentage-point recall@8 override.
All selected width-12 candidates had observed transition support, so
`observed_support` tied `all_ranked` and won the predefined simplicity tie-break.

The immutable selection payload SHA-256 is
`a44de68e4904f65f8cd5e8f4594cd3f35d72f3c070107941573d00da9d4c576b`.
Before test access, its lock recorded `test_opened=false`.

## Single sealed-test result

Selected model: source-normalized, phase-separated B2 transition predictor.
Artifact size is 2,339,899 bytes; SHA-256 is
`6650e2055b9b5762d0a8d410cd712dd1b54a5db90e0e6ef43ccb63ecff1e1dc4`.

| Metric | Result |
|---|---:|
| Samples | 28,043 |
| Recall@8 | 55.1060% |
| Recall@12 | 67.3443% |
| Recall@16 | 74.8404% |
| Mean overlap@8 | 4.4085 experts |
| Exact top-8 set match | 4.1650% |
| CPU latency p50 / p95 | 76.7 / 83.6 us |

Recall@12 by domain:

| Domain | Samples | Recall@12 |
|---|---:|---:|
| Code | 4,495 | 70.97% |
| General instruction | 4,437 | 69.54% |
| Math/reasoning | 4,727 | 66.90% |
| Structured output | 5,220 | 65.96% |
| Topic shift | 4,466 | 68.62% |
| Translation/multilingual | 4,698 | 62.57% |

Prefill recall@12 is 70.02%; decode recall@12 is 63.06%. Per-layer and
per-conversation results are preserved in `test-metrics.json`; layer recall@12
ranges from 57.83% at target layer 2 to 71.41% at target layer 28.

## Simulated 32-slot shadow

The timing model assumes every prediction is ready. It models neither CUDA
events nor copy/compute overlap.

| Metric | Result |
|---|---:|
| Expert demands | 224,344 |
| Reactive hits / misses | 164,252 / 60,092 |
| Ready demands | 187,969 |
| Ready improvement | 23,717 |
| Uncovered misses | 36,375 |
| Miss reduction | 39.47% |
| Eviction regret | 6,099 |
| Net gain after regret | 17,618 |
| Useful / wasted insertions | 27,919 / 41,868 |
| Useful / wasted projected bytes | 93.40 GB / 140.07 GB |
| Additional speculative evictions | 59,635 |

The selected observed-support rule rejected zero width-12 candidates on test,
because all ranked candidates in the selected prefix had positive training
support.

## Commands and seal verification

```powershell
$env:PYTHONPATH='src;.'
python scripts\run_next_layer_predictor.py fit `
  --manifest C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json `
  --output C:\models\expertflow\runs\expanded-predictor-final --expanded

python scripts\run_next_layer_predictor.py test `
  --manifest C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json `
  --output C:\models\expertflow\runs\expanded-predictor-final --expanded
```

The fit took 295.7 seconds and the sole test evaluation took 29.6 seconds.
An intentional second test invocation failed with `expanded test split has
already been evaluated`; the existing test-metrics hash remained unchanged.

External artifacts are under
`C:\models\expertflow\runs\expanded-predictor-final`. The complete command
ledger, model hashes, validation metrics, selection lock, test metrics, and
artifact index are preserved there.
