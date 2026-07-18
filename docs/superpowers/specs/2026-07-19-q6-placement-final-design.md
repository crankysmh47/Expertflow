# Q6 Placement Optimizer Final Sprint Design

## Goal

Find the highest-throughput full-capacity static Q6 expert placement that fits the RTX 5060 Ti with a 256 MiB reserve, then stop unless the measured configuration reaches the product threshold of 23.5 decode TPS and beats the strongest fair stock result of 22.967 TPS.

## Architecture

The existing synchronized split probe remains a disabled diagnostic facility. An offline ranking tool converts repeated stock profiles into per-layer CPU expert time, selected-path share, full-shadow MiB, and time-per-MiB. The runtime change is deliberately limited to increasing the existing fixed static-island layer/shadow capacities from four layers/sixteen tensors to twelve layers/forty-eight tensors; all allocation, exact packed copying, identity mapping, graph binding, and teardown semantics remain unchanged.

Search uses the frozen four-layer set `[0,1,15,20]`, then tests two-layer additions in ranked order. Each candidate receives two alternating cold-process smoke pairs with graphs explicitly set, 512 generated tokens, process-owned VRAM sampling, deterministic response hashes, and a short quality smoke. A pair is retained only when it adds at least 0.25 TPS. Two consecutive failures, insufficient margin, a quality breach, or an architectural dependency terminates expansion.

## Boundaries

- No cache, eviction, transfer-on-demand, prediction, new CUDA operation, scheduler redesign, or whole-layer placement.
- No measured run enables split profiling, observer tracing, or debug callbacks.
- Static source tensors remain CPU authoritative; every CUDA shadow contains all 128 packed experts and the matching scale tensor.
- Candidate sets are passed as a single startup environment value and parsed once per context.
- Runtime tuning is sequential and one-variable-at-a-time. Stock and ExpertFlow may retain different fastest settings, but neither is intentionally handicapped.
- Held-out WikiText-103 and frozen MMLU run only for the final provisional winner.
- Product commands are built only after a validated ExpertFlow result reaches 23.5 TPS and beats stock.

## Evidence and decisions

All profile repetitions, rankings, candidates, rejected pairs, commands, hashes, temperatures, clocks, VRAM, output identities, quality results, and stop reasons are serialized under `docs/evidence/q6-placement-final/`. Large raw logs remain outside Git under `C:\models\expertflow\runs\q6-placement-final` and are referenced by SHA-256.

If the first expansion crosses 23.5 TPS, search stops and moves to bounded cleanup/tuning. If the best stable static result remains below 22.967 TPS or two consecutive additions fail the marginal rule, the final verdict is `STATIC EXPANSION STOP` and dynamic/product stages remain closed.

## Design self-review

The design covers the requested static search, fast-path/tuning gates, independent stock comparison, final quality validation, packaging gate, evidence layout, and terminal conditions. It contains no dynamic-cache implementation assumption and no ambiguous fallback that would broaden runtime architecture.
