# ExpertFlow Q6 predictive follow-up

## Result

The frozen static champion is valid and MMLU-supportive, but the bounded hybrid cache gate fails. The work stops at **NO CACHE OPPORTUNITY** without changing llama.cpp, implementing reactive Q6 caching, integrating prediction, or packaging cached product profiles.

## Static quality

The fixed 100-item protocol scored 49/100 OFF and 50/100 ON, a +1.0 percentage-point change. Fourteen answers changed: five OFF-to-ON improvements, four regressions, and five changes where neither answer was correct. Every changed ON item repeated exactly in item identity, prediction, token IDs, and content. No CUDA or NaN error occurred and all processes cleaned up.

This is `MMLU SUPPORTIVE`. It does not alter the frozen PPL result: the point estimate improved by 2.92%, but the paired 95% upper bound remains +2.25%, so the original strict PPL gate remains failed.

## Q6 routing evidence

The cache-disabled canonical observer generated 127 fresh Q6 shards and 655,740 events with all 30 routed layers and exactly eight selected experts per event:

- 512-token performance workload: 1 shard, 16,170 events.
- Eight evenly spaced 2,048-character windows from the frozen held-out PPL corpus: 118,500 events. These are routing samples from the corpus, not a rerun or reinterpretation of PPL.
- Fixed MMLU: 100 independently reset shards, 477,210 events.
- Expanded representative corpus: 18 independently reset conversations balanced across six domains and train/validation/test, 43,860 events.

The Q6 expert bundle is 5,358,852 bytes. `routing-locality.json` reports frequency, per-sequence working sets, reuse-distance distributions, temporal and adjacent-token reuse, and reset-LRU results at 112/96/80/64 slots for every layer and workload family. Existing predictors are reported unavailable for Q6 because their frozen training/evaluation evidence is Q4-only.

## Hybrid simulation gate

The simulation uses measured Q6 miss rates, exact Q6 bundle bytes, the measured 4,768.7 MiB/s effective Q4 live-cache copy rate scaled by bytes, and the measured 9.96% nine-layer Q4 residual cache-path cost apportioned per cached layer. Transfer and remapping costs are explicitly projections, not Q6 runtime measurements.

The first candidate meeting the 500 MiB memory requirement keeps eight layers static and caches `[20,15,9,5]` at 96 slots. It frees 654.16 MiB but projects 26.35 TPS, retaining 93.66% of 28.13—below the required 26.72 TPS/95%. Six cached layers free 981.24 MiB and permit one additional full layer, but project 25.50 TPS, or only 25.83 TPS with the optimistic added-layer gain.

No candidate passes both gates. Per the frozen plan, this prevents Stage 4 source modification and every later reactive, predictor, policy, context/concurrency, and product experiment.

## Preservation

The frozen static champion remains 28.13 TPS at 10,966.801 MiB with layers `[0,1,2,3,4,5,6,7,8,9,15,20]`; strongest stock remains 22.967 TPS. Both new branches remain isolated, unmerged and unpushed. The llama.cpp follow-up branch has no modifications.
