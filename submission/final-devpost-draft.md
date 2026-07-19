# ExpertFlow: profile-guided MoE deployment on a 16 GB GPU

Gemma 4 26B A4B is sparse, but its expert banks still have to live somewhere. Stock llama.cpp can keep them on the CPU, which fits, or move whole layers to CUDA, which is too coarse for this memory budget. ExpertFlow profiles the routed-expert work separately and places only the expert banks from the most valuable layers on CUDA.

The winning Q6 deployment uses full 128-expert static banks for layers 0–9, 15, and 20. It has no reactive cache, eviction, predictor, or per-token expert transfer. On an RTX 5060 Ti 16 GB, ten matched 512-token runs measured 28.13 decode TPS. The strongest stock result was 22.967 TPS, so the measured improvement is 22.48%. Peak process-owned VRAM was 10,966.801 MiB.

We also tested the uncomfortable part: quality. MMLU moved from 49/100 to 50/100, and all fourteen changed ON answers repeated exactly. The perplexity point estimate improved by 2.92%, but the 95% upper bound was +2.25%. Our strict +1% confidence requirement did not pass. We report both results.

Predictive caching looked attractive early on, then lost the measurement argument. A bounded simulation combined measured Q6 routing with measured Q4 cache costs and found no candidate that saved enough memory without giving back too much throughput. The product therefore ships full static residency on this GPU.

The release wraps the result in a small CLI. Judges can replay and hash-check the evidence without a GPU, inspect a compatible Windows/CUDA system, generate a deployment, run the benchmark, start an OpenAI-compatible server, or compare recorded results. A five-repetition live server test measured 35.67 aggregate generated TPS with four slots versus 24.52 stock. A bounded context test allocated the model's 262,144-token training context, though only 417 tokens were processed, so it is not a filled-context claim.

ExpertFlow is intentionally narrow. It is a working deployment optimizer for this verified Gemma Q6 case, with a manifest interface that can support more models after they earn their own evidence.

