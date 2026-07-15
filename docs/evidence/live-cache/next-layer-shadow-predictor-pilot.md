# Next-Layer Shadow Predictor Pilot

## Verdict

**PROMISING SMALL-PILOT FEASIBILITY.** Training-only layer-to-layer transition counts (B2) beat the training-only target-layer frequency baseline (B1) on the frozen four-conversation validation split, generalized to the three sealed test conversations, and improved simulated ready coverage over the identical reactive 32-slot LRU. This is offline shadow evidence only; no runtime transfer, overlap, or speedup is claimed.

The immutable canonical split remains seven train / four validation / three test. The test split was opened only after `selection-lock.json` froze B2, seed `20260716`, the binary feature contract, widths 8/12/16, and width 8 as the recommended candidate width.

## Validation selection

| Model | Recall@8 | Recall@12 | Recall@16 | Exact set@8 | Batch-1 CPU p50 | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| B0 current-set copy | 6.32% | 9.43% | 12.68% | 0.00% | 13.9 us | 0 |
| B1 target-layer frequency | 23.35% | 31.02% | 37.59% | 0.12% | 11.7 us | 0 |
| **B2 transition** | **41.66%** | **52.75%** | **60.99%** | **0.46%** | 50.6 us | 0 |
| B3 linear | 38.63% | 48.85% | 56.40% | 0.17% | 5.8 us | 36,864 |
| B4 shared MLP | 36.29% | 46.04% | 53.30% | 0.13% | 13.6 us | 26,752 |

Canonical traces contain no reliable routing weights, so the current selection is encoded as binary values. B3/B4 add target-layer one-hot identity, phase, and a uniquely joined causal previous-token target-layer vector. B4 is the only attempted MLP architecture.

## Sealed test result

B2 evaluated 13,427 adjacent-layer samples:

- recall@8: **42.98%**
- recall@12: **54.48%**
- recall@16: **62.84%**
- mean overlap@8: **3.439 of 8**
- exact-set@8: **0.48%**
- batch-one CPU latency: **45.9 us p50 / 57.4 us p95**
- decode recall@8/16: 36.39% / 55.73%
- prefill recall@8/16: 44.59% / 64.57%

The frozen model comparison was also evaluated without changing any configuration. Test recall@8/12/16 is 6.68%/9.52%/12.92% for B0, 27.49%/36.03%/42.91% for B1, 42.98%/54.48%/62.84% for B2, 39.38%/49.76%/57.39% for B3, and 37.40%/47.39%/54.72% for B4. B2 therefore beats frequency and both learned models on validation and final test.

Per-conversation recall@8/16 is 47.00%/67.42% for multilingual, 42.19%/61.83% for topic shift, and 40.79%/60.50% for structured output. Full per-layer, phase, and conversation breakdowns are in `test-metrics.json`.

## Simulated 32-slot shadow cache

At locked width 8, test demand contains 107,416 expert selections. Reactive LRU has 78,442 ready hits and 28,974 blocking misses. B2 shadow insertion produces 82,204 ready demands, a simulated gain of **3,762** and a **12.98%** reduction in uncovered misses to 25,212.

The trade-off is material: 4,760 useful speculative insertions versus 10,260 wasted insertions, 13,613 speculative evictions, and 1,338 eviction-regret events. Projected packed transfer is 15.92 GB useful and 34.32 GB wasted. Width 12 gains 5,030 ready demands but increases waste to 67.81 GB and regret to 2,595; width 16 gains 5,529 but wastes 117.16 GB with 4,416 regret events. Width 8 is therefore retained for later live shadow measurement.

The simulator assumes every prediction is ready before the next layer and reports zero late predictions. It does not model transfer/compute timing or overlap.

## Artifacts and limits

Artifacts are under `C:\models\expertflow\runs\next-layer-shadow-predictor-pilot`. The selected B2 artifact is 870,078 bytes. `validation-metrics.json`, `selection-lock.json`, `test-metrics.json`, and `ledger.jsonl` preserve the full evidence and frozen decision sequence.

Fourteen synthetic conversations cannot support broad generalization. The result establishes structured next-layer routing and a plausible shadow-prefetch candidate only. Integration remains prohibited until C5 passes and runtime shadow latency/usefulness are directly measured.
