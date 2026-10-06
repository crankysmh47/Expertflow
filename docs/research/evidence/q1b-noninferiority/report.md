# ExpertFlow Q1b non-inferiority and layer sensitivity

## Verdict

`Q1B PASS`

Q1b passes its independently frozen product-quality gates. This does **not** revise Q1: the official Q1 result remains a failure at `+0.586365%` against its original `+0.500000%` threshold.

Q1b is a quality pass, not a speedup result. The best four-layer configuration improved held-out perplexity and was exactly deterministic, but its measured decode-TPS change was only `+0.49%`, smaller than run-to-run dispersion. Off/on generated behavior was not identical across the broader suite.

## Frozen-gate results

| Gate | Reference | Candidate | Result |
|---|---:|---:|---|
| Independent WikiText-103 validation PPL | 735.7794 | 726.0783 | `-1.3185%`; PASS |
| Paired block-bootstrap 95% interval | - | `[-3.2017%, +0.5675%]` | upper `<= +1.0%`; PASS |
| MMLU, frozen 100 questions | 42/100 | 41/100 | `-1.0 pp`; PASS at inclusive boundary |
| Candidate repeat determinism | - | 100/100 item outputs identical | PASS |
| Two-layer `[0,15]` PPL | 735.7794 | 712.9249 | `-3.1062%`; PASS |
| Four-layer `[0,1,15,20]` PPL | 735.7794 | 710.7986 | `-3.3951%`; PASS |
| Four-layer stability | - | three exact repetitions | PASS |

The independent PPL comparison contains 8,184 paired scored tokens. Bootstrap settings were frozen at 10,000 samples, seed `20260718`, and contiguous 128-token blocks within each 2,048-token chunk. Mean NLL changed from `6.6009303188` to `6.5876579029`.

## Measurement validation

The measurement-only NLL sidecar records the target token's already-computed post-logit negative log-likelihood. It does not change logits, graph placement, execution order, or CUDA scheduling. A tiny-model contract produced exactly 30 ordered records, and the Q1 reproduction produced all 4,092 expected records per mode.

Q1 reproduced exactly:

| Mode | Original Q1 | Q1b reproduction |
|---|---:|---:|
| Feature off | 1176.7406 | 1176.7406 |
| Layer 0 on | 1183.6406 | 1183.6406 |
| Relative change | `+0.586365%` | `+0.586368%` from token-level NLL |

The post-scaling-extension disabled binary was also compared against the frozen reference for 2,046 scored tokens; every JSONL line was bit-identical.

## MMLU

The fixed evaluation used ten frozen subjects with ten questions each, a zero-shot prompt, temperature 0, seed 42, prompt caching disabled, and a grammar-constrained single `A-D` token. The reference scored 42%; layer 0 scored 41%. A complete candidate repeat matched every item identity, predicted option, generated token ID, and content.

Subject results are intentionally treated as small-sample diagnostics. Individual ten-item subject swings ranged from `-20 pp` to `+20 pp`; they are not broad subject-level claims. All 23 off/on disagreements are recorded in `mmlu-results.json`.

## Individual-layer sensitivity

Layers 0, 15, and 20 represent an early layer, the midpoint, and the latest CPU-resident routed layer at matched `-ngl 10`. Layer 20 was chosen instead of an already CUDA-resident layer to avoid changing MoE placement or measuring a hidden duplicate.

| Layer | PPL change | Descriptive 95% interval | Decode TPS | TPS change vs 27.0667 | Token agreement | Router set/order overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `-1.3185%` | `[-3.2017%, +0.5675%]` | 27.2 | `+0.49%` | 6/7 | 70.87% / 46.87% |
| 15 | `+0.8791%` | `[-0.2001%, +2.0295%]` | 26.2 | `-3.20%` | 5/7 | 77.63% / 63.85% |
| 20 | `+0.2103%` | `[-0.6362%, +1.0566%]` | 26.8 | `-0.99%` | 5/7 | 78.65% / 67.49% |

The required ranking unit is decode-TPS percent change per positive PPL-regression percentage point. Layer 0 is dominant because quality improved and its measured TPS did not decrease. Layer 15's descriptive ratio is `-3.64`; layer 20's is `-4.69`. Both ratios have negative numerators, so neither is evidence of acceleration, and single-run individual TPS values are diagnostic only.

Router overlap is a behavioral-sensitivity measurement, not a correctness failure: early numerical changes alter later hidden states and therefore later routing. Q1b demonstrates deterministic aggregate quality, not off/on token identity.

## Scaling and performance

Each complete Q4 expert bundle is exactly 428,212,736 bytes: gate/up 285,474,816 bytes, down 142,737,408 bytes, and scale 512 bytes.

| Layers | Exact arena | PPL change | 95% interval | Decode TPS |
|---|---:|---:|---:|---:|
| `[0]` | 408.3755 MiB | `-1.3185%` | `[-3.2017%, +0.5675%]` | 27.2 |
| `[0,15]` | 816.7510 MiB | `-3.1062%` | `[-4.8717%, -1.3171%]` | 26.6 |
| `[0,1,15,20]` | 1,633.5020 MiB | `-3.3951%` | `[-6.0301%, -0.8264%]` | 27.4, 26.9, 27.3 |

Matched 128-token CLI measurements:

- Feature off decode TPS: `26.8, 27.1, 27.3`; mean `27.0667`, sample variance `0.0633`.
- Four layers decode TPS: `27.4, 26.9, 27.3`; mean `27.2`, sample variance `0.0700`.
- Relative decode change: `+0.4926%`.
- Feature off prompt TPS mean: `24.0667`; four layers: `25.9` (`+7.62%`).
- Generation-suite peak system GPU use: 6,681 MiB off and 8,329 MiB for four layers. The separate PPL run peaked at 8,386 MiB.

The four-layer canonical suite used seven fixed general, code, arithmetic, structured-output, translation, and reproducibility prompts. Across three repetitions, prompt tokens, generated tokens, and every normalized router event were exact between repetitions. All 21 best-four processes exited cleanly and GPU use returned to baseline. Compared with feature off, generated tokens matched on 4/7 prompts; aggregate router top-k set/order overlap was 57.84%/34.56%.

No NaN, CUDA error, invalid memory access symptom, persistent GPU growth, or failed context teardown was observed. `compute-sanitizer` was unavailable in the installed supported Windows CUDA toolchain, so sanitizer status is explicitly `not_run`, not passing.

## Recommendation

The recommended **quality-safe selective candidate set** is `[0,1,15,20]`. Its measured quality cost is `-3.3951%` PPL (an improvement), its measured decode change is only `+0.4926%`, and it uses 1,712,850,944 exact arena bytes.

Do not claim a runtime speedup and do not begin reactive caching from Q1b alone. The core product objective is usable end-to-end acceleration; this study shows quality headroom but no meaningful throughput gain.

Q6 applicability remains unresolved. The preserved Q6 placement blocker was not changed, Q6 bundle movement was not implemented, and the Q4/Q6 TPS values are not directly interchangeable. There is currently no credible measured basis that selective reactive Q6 caching will beat the fair `22.967 TPS` Q6 baseline.

Exact next step, if runtime work is resumed: first reproduce the `[0,1,15,20]` static selective placement with Q6 packed bundles behind a disabled-by-default flag, then run at least ten matched cold-process off/on repetitions with exact output checks and a predeclared TPS threshold. Stop before reactive loading or prediction unless Q6 placement is exact and its confidence interval shows a material end-to-end gain. For the present submission, freeze this Q1b milestone and proceed with the protected release/Observatory evidence.

## Evidence map

- `results.json`: machine-readable gate and scaling summary.
- `corpus-manifest.json`: immutable corpus/runtime identities.
- `per-token-nll-off.jsonl`, `per-token-nll-on.jsonl`: independent paired NLL records.
- `mmlu-results.json`: aggregate, subject, disagreement, and repeat data.
- `layer-sensitivity.json`: individual layer quality, routing, TPS, VRAM, and ranking.
- `generation-performance-summary.json`: prompt-level token/router parity, repetitions, and performance.
- `raw/`, `generation/`, `performance/`: command outputs and measured manifests.
