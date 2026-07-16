# ExpertFlow Runtime Freeze

Effective 2026-07-16, runtime experimentation is frozen after the final
projected-state temporal-policy benchmark.

## Included milestones

- exact one-layer eight-slot packed execution
- exact layer-24 32-slot reactive LRU
- exact CUDA-resident multi-layer cache evidence
- frozen offline and live-shadow predictors
- exact two-slot asynchronous temporal sidecar
- final projected post-admission candidate filter

## Release defaults

- `live_cache_enabled=false`
- predictive shadow disabled
- temporal shadow disabled
- asynchronous sidecar disabled
- Observatory and judge replay remain the supported submission path

## Closed work

Do not begin another predictor, predictor retuning, multi-layer prefetch, MTP,
RL, wider transfer concurrency, higher offload exploration, or cache-size
sweep. Any future runtime work must occur after the hackathon release as a new,
separately authorized research stage.

## Release integration rule

Preserve exactness evidence and negative performance results. Do not claim a
runtime speedup. Clearly separate measured live-runtime results from standalone
CUDA measurements, simulations, projections, and oracle analysis.
