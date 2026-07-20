# Claims ledger

| Claim | Class | Source and qualification |
|---|---|---|
| Gemma 4 26B A4B Q6 ran at 28.13 TPS with twelve static expert banks. | Measured | `docs/evidence/q6-placement-final/results.json`; ten matched 512-token runs. |
| The result is 22.48% faster than the strongest stock result of 22.967 TPS. | Measured | Same evidence; separate strongest-stock protocol. |
| Peak process-owned VRAM was 10,966.801 MiB. | Measured | Same evidence. |
| MMLU moved from 49/100 to 50/100. | Measured | `docs/evidence/q6-predictive-final/mmlu-static.json`; 14 changed ON answers repeated exactly. |
| PPL point estimate changed by -2.92%; 95% upper bound was +2.25%. | Measured | `docs/evidence/q6-placement-final/quality-results.json`. Strict 1% perplexity confidence gate was not met. |
| Four-slot server throughput was 35.6699 aggregate generated TPS versus 24.5231 stock. | Measured | `docs/evidence/product-release/throughput-profile.json`; five cold servers per mode, 20/20 requests. Concurrent outputs were not fully deterministic. |
| The context profile allocated 262,144 tokens and processed 417. | Measured | `docs/evidence/product-release/context-profile.json`; full capacity was not filled. |
| Predictive caching has `NO CACHE OPPORTUNITY` on this configuration. | Simulated | `docs/evidence/q6-predictive-final/cache-simulation.json`; measured Q6 routing with measured Q4 cache costs. It is not a live Q6 cache result. |
| A 96-slot hybrid would run at 26.35 TPS. | Projected | Rejected simulation candidate. This number is never presented as measured. |
| Wider hardware and model support can use the same manifest interface. | Planned | Extension interface exists; no broad compatibility claim. |

