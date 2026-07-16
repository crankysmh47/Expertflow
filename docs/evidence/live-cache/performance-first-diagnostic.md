# Performance-First Diagnostic Benchmark

## Verdict

**PASS — PROCEED TO THE STAGED MULTI-LAYER RAMP.** C5 preserved exact cache-off/cache-on tokens and routing, reduced layer-24 misses by **57.33%**, reduced aggregate measured layer-24 blocking host-wall time by **45.45%**, and improved mean decode TPS by **9.75%** versus the canonical observer/cache-off runtime. Its allocation and cleanup were stable, and no severe unexplained bookkeeping regression appeared.

This does **not** establish the final product claim. The strongest stable stock configuration, full offload at `-ngl 99`, remains much faster: **98.53 decode TPS** versus C5's **27.71 TPS**. C5 is 71.88% behind that reproduced same-hardware baseline. Broader layer coverage is therefore necessary before ExpertFlow can claim usable end-to-end superiority.

## Fixed protocol

- model: Gemma 4 26B-A4B Q4_0, SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- prompts: fixed general, code, and French-translation prompts
- sampling: greedy; 64 requested generated tokens; early EOG retained
- context execution: identical probe, batch/ubatch 1, 12 CPU threads
- repetitions: one discarded warmup plus three measured runs per prompt and mode
- modes: stock best `-ngl 99`, stock matched `-ngl 10`, observer/cache-off `-ngl 10`, C4 eight-slot `-ngl 10`, C5 32-slot `-ngl 10`
- timing: host wall around prompt/decode phases plus independent llama.cpp counters
- token latency: host-wall intervals already captured by the benchmark probe
- GPU peak: system-wide NVIDIA memory samples at 100 ms, not process-owned allocation
- cache duration: blocking host wall around copies and synchronization; **not** CUDA-event latency and **not** copy/compute overlap

The benchmark-only probe linked the unchanged verified DLL sets. No llama.cpp source was modified.

Benchmark probe SHA-256 values:

- stock: `edd5e341325c85e3abe264f9d28821f36541435c2b29259a3b36fb72f4b900f3`
- C4: `9caad32d6a7bc17a5303d659e414426eebc0a3977d044cdde41e57efa4843645`
- C5/observer: `1432da116795a31a36a20eb118fa36d5ace8c716694740cde1afffd43892eb97`
- machine summary: `89997f3d11eb3efeaf73a721685dc1e66a570e5d9d3a96c3af92fa9d4c69663c`

## Stock offload selection

One diagnostic general-prompt run was made at each bounded candidate. All candidates were stable and exited normally.

| `-ngl` | Decode TPS | End-to-end ms | Settled GPU before/after MiB |
|---:|---:|---:|---:|
| 0 | 15.98 | 7,270.9 | 2,213 / 2,212 |
| 5 | 20.31 | 5,572.7 | 2,212 / 2,222 |
| 10 | 25.40 | 4,571.2 | not sampled in the initial smoke |
| 15 | 31.43 | 3,675.9 | 2,222 / 2,209 |
| 20 | 40.54 | 2,856.7 | 2,209 / 2,212 |
| 25 | 54.73 | 2,091.2 | 2,212 / 2,211 |
| 30 | 83.54 | 1,334.0 | 2,211 / 2,130 |
| 99 | **98.69** | **1,126.9** | 2,130 / 1,730 |

`-ngl 99` was selected by highest measured decode TPS, with end-to-end time as tie-break. The full matrix confirmed stability at a mean system-wide peak of 15,694.7 MiB on the 16,311 MiB GPU, leaving about 616 MiB at the sampled peak.

## Overall results

Means and sample variances below pool the nine measured runs per mode. Prompt lengths differ by domain, so the domain table is the primary detailed view.

| Mode | Prompt TPS mean (var) | Decode TPS mean (var) | E2E s mean (var) | TTFT s mean (var) | Token p50 ms mean (var) | Token p95 ms mean (var) | System GPU peak MiB mean (var) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock best `-ngl 99` | 91.60 (0.93) | **98.53 (18.29)** | **1.080 (0.019)** | **0.560 (0.004)** | **9.95 (0.35)** | **11.13 (0.05)** | 15,694.7 (2.5) |
| Stock matched `-ngl 10` | 22.68 (0.28) | 26.50 (1.22) | 4.208 (0.325) | 2.220 (0.056) | 37.90 (1.79) | 42.83 (9.84) | 6,460.9 (372.9) |
| Observer/cache-off | 21.33 (1.21) | 25.25 (0.47) | 4.439 (0.434) | 2.358 (0.060) | 40.13 (0.69) | 45.24 (6.94) | 6,438.6 (7.8) |
| C4 eight-slot | 20.61 (2.41) | 25.75 (2.09) | 4.500 (0.450) | 2.434 (0.019) | 39.57 (3.10) | 43.52 (7.78) | 6,080.0 (21.0) |
| C5 32-slot LRU | 22.04 (0.22) | **27.71 (0.79)** | **4.183 (0.300)** | **2.280 (0.062)** | **36.72 (0.95)** | **39.95 (2.92)** | 6,211.1 (1.1) |

System-wide GPU peaks include the desktop and varied with unrelated applications; they must not be interpreted as exact runtime allocations. The exact measured persistent cache arenas remain 25.523 MiB for C4 and 102.094 MiB for C5.

## Repetition-level domain results

Each cell is `rep1 / rep2 / rep3 → mean (sample variance)`. Counts are exact and constant within each domain.

| Mode/domain | Prompt / generated tokens | Prompt TPS | Decode TPS | E2E seconds | TTFT seconds | p50 ms | p95 ms | System peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stock best/general | 42 / 64 | 90.84/90.80/90.96 → 90.87 (0.01) | 98.27/98.44/95.52 → 97.41 (2.68) | 1.114/1.113/1.132 → 1.119 (0.000) | 0.473/0.472/0.472 → 0.472 (0.000) | 10.06/9.97/10.32 → 10.11 (0.03) | 10.97/10.88/11.28 → 11.04 (0.04) | 15693/15696/15696 → 15695 (3) |
| Stock best/code | 53 / 64 | 92.60/92.49/90.76 → 91.95 (1.06) | 95.93/109.45/96.72 → 100.70 (57.55) | 1.240/1.158/1.246 → 1.214 (0.002) | 0.583/0.583/0.595 → 0.587 (0.000) | 10.31/8.40/10.18 → 9.63 (1.14) | 11.09/11.04/11.45 → 11.19 (0.05) | 15696/15696/15696 → 15696 (0) |
| Stock best/translation | 56 / 29 | 91.88/93.25/90.86 → 91.99 (1.44) | 98.50/95.73/98.26 → 97.49 (2.36) | 0.904/0.904/0.912 → 0.906 (0.000) | 0.619/0.612/0.626 → 0.619 (0.000) | 10.09/10.20/10.02 → 10.10 (0.01) | 10.89/11.47/11.11 → 11.16 (0.09) | 15693/15693/15693 → 15693 (0) |
| Stock 10/general | 42 / 64 | 21.75/22.17/22.12 → 22.01 (0.05) | 27.16/25.86/25.07 → 26.03 (1.11) | 4.288/4.369/4.451 → 4.369 (0.007) | 1.937/1.900/1.904 → 1.913 (0.000) | 36.81/38.86/39.27 → 38.31 (1.74) | 40.93/43.13/46.18 → 43.41 (6.96) | 6470/6470/6485 → 6475 (75) |
| Stock 10/code | 53 / 64 | 23.14/23.07/22.83 → 23.01 (0.03) | 24.71/27.13/26.22 → 26.02 (1.50) | 4.881/4.656/4.762 → 4.766 (0.013) | 2.296/2.302/2.327 → 2.308 (0.000) | 40.34/36.56/37.82 → 38.24 (3.70) | 48.79/41.28/44.34 → 44.80 (14.26) | 6470/6476/6469 → 6472 (14) |
| Stock 10/translation | 56 / 29 | 22.80/23.06/23.20 → 23.02 (0.04) | 27.88/27.50/27.01 → 27.46 (0.19) | 3.496/3.483/3.488 → 3.489 (0.000) | 2.462/2.433/2.419 → 2.438 (0.000) | 36.45/37.21/37.76 → 37.14 (0.43) | 39.34/39.59/41.89 → 40.27 (1.97) | 6436/6436/6436 → 6436 (0) |
| Observer/general | 42 / 64 | 21.58/19.10/21.05 → 20.57 (1.71) | 25.37/25.40/25.55 → 25.44 (0.01) | 4.469/4.719/4.501 → 4.563 (0.019) | 1.947/2.200/1.996 → 2.048 (0.018) | 39.76/40.01/40.02 → 39.93 (0.02) | 45.12/43.04/42.86 → 43.67 (1.58) | 6443/6439/6437 → 6440 (9) |
| Observer/code | 53 / 64 | 20.33/21.09/21.58 → 21.00 (0.39) | 24.72/24.45/24.70 → 24.62 (0.02) | 5.196/5.132/5.047 → 5.125 (0.006) | 2.607/2.514/2.457 → 2.526 (0.006) | 40.83/41.12/40.17 → 40.71 (0.24) | 45.62/45.50/50.66 → 47.26 (8.67) | 6443/6435/6436 → 6438 (19) |
| Observer/translation | 56 / 29 | 22.44/22.35/22.40 → 22.40 (0.00) | 26.13/24.57/26.35 → 25.68 (0.94) | 3.605/3.685/3.600 → 3.630 (0.002) | 2.496/2.506/2.500 → 2.501 (0.000) | 39.14/41.27/38.87 → 39.76 (1.73) | 44.50/47.61/42.20 → 44.77 (7.38) | 6438/6438/6438 → 6438 (0) |
| C4/general | 42 / 64 | 18.36/19.13/18.31 → 18.60 (0.21) | 23.91/23.61/25.51 → 24.34 (1.04) | 4.963/4.907/4.803 → 4.891 (0.007) | 2.288/2.196/2.295 → 2.260 (0.003) | 41.70/42.74/39.71 → 41.38 (2.36) | 47.04/48.40/43.10 → 46.18 (7.56) | 6072/6078/6090 → 6080 (84) |
| C4/code | 53 / 64 | 21.41/21.21/21.27 → 21.30 (0.01) | 25.54/25.39/25.53 → 25.49 (0.01) | 4.981/5.019/4.999 → 5.000 (0.000) | 2.476/2.499/2.493 → 2.489 (0.000) | 39.31/39.65/39.62 → 39.53 (0.04) | 44.00/44.25/42.58 → 43.61 (0.81) | 6080/6080/6080 → 6080 (0) |
| C4/translation | 56 / 29 | 21.86/21.94/22.02 → 21.94 (0.01) | 27.52/27.65/27.06 → 27.41 (0.10) | 3.616/3.601/3.614 → 3.610 (0.000) | 2.563/2.553/2.543 → 2.553 (0.000) | 37.46/37.48/38.43 → 37.79 (0.31) | 40.68/39.99/41.60 → 40.76 (0.66) | 6080/6080/6080 → 6080 (0) |
| C5/general | 42 / 64 | 21.35/21.63/21.39 → 21.46 (0.02) | 27.97/27.94/27.50 → 27.81 (0.07) | 4.255/4.233/4.290 → 4.259 (0.001) | 1.968/1.943/1.964 → 1.958 (0.000) | 35.95/36.11/37.04 → 36.36 (0.35) | 39.70/38.86/39.78 → 39.45 (0.26) | 6210/6210/6210 → 6210 (0) |
| C5/code | 53 / 64 | 22.37/22.14/22.35 → 22.28 (0.02) | 27.46/26.80/25.97 → 26.75 (0.56) | 4.700/4.782/4.836 → 4.773 (0.005) | 2.370/2.395/2.373 → 2.379 (0.000) | 36.86/37.75/38.68 → 37.76 (0.82) | 40.15/41.78/43.51 → 41.81 (2.83) | 6210/6212/6212 → 6211 (1) |
| C5/translation | 56 / 29 | 22.37/22.67/22.10 → 22.38 (0.08) | 28.51/28.75/28.47 → 28.58 (0.02) | 3.521/3.479/3.553 → 3.518 (0.001) | 2.504/2.471/2.535 → 2.504 (0.001) | 36.12/35.68/36.28 → 36.03 (0.10) | 38.82/38.83/38.09 → 38.58 (0.18) | 6212/6212/6212 → 6212 (0) |

## Cache accounting

Counts and bytes were deterministic across all three repetitions. Blocking values are rep1/rep2/rep3 host-wall milliseconds.

| Mode/domain | Expert demands | Hits | Misses | Hit rate | Bytes transferred | Aggregate blocking wall ms | All blocking / generated token |
|---|---:|---:|---:|---:|---:|---:|---:|
| C4/general | 840 | 326 | 514 | 38.81% | 1,719,541,768 | 364.327 / 360.342 / 340.836 | 5.55 ms |
| C4/code | 928 | 272 | 656 | 29.31% | 2,194,590,272 | 405.633 / 408.158 / 405.395 | 6.35 ms |
| C4/translation | 672 | 253 | 419 | 37.65% | 1,401,727,628 | 278.631 / 277.121 / 277.686 | 9.58 ms |
| C5/general | 840 | 658 | 182 | 78.33% | 608,864,984 | 166.981 / 165.746 / 167.790 | 2.61 ms |
| C5/code | 928 | 632 | 296 | 68.10% | 990,241,952 | 228.185 / 228.743 / 227.946 | 3.57 ms |
| C5/translation | 672 | 472 | 200 | 70.24% | 669,082,400 | 172.852 / 170.529 / 172.060 | 5.93 ms |

Across the nine measured runs:

- C4: mean 813.33 demands, 283.67 hits, 529.67 misses, 34.88% hit rate, 1,771,953,222.67 bytes, and 346.459 ms aggregate blocking per run.
- C5: mean 813.33 demands, 587.33 hits, 226.00 misses, 72.21% hit rate, 756,063,112 bytes, and 188.981 ms aggregate blocking per run.
- Normalizing all prompt-plus-decode layer-24 blocking by generated outputs gives 6.62 ms/token for C4 and 3.61 ms/token for C5.
- Splitting by token index, decode-phase blocking alone averaged 127.529 ms/run (2.307 ms/generated token) for C4 and 40.881 ms/run (0.778 ms/generated token) for C5.

Cache metadata/remapping cost is not separately instrumented. The non-copy residual cannot be identified exactly because cache logging, synchronization, scheduling, CUDA graph behavior, and ordinary execution share the same host-wall interval. The end-to-end and decode comparisons below bound their combined effect.

## Explicit comparisons

- Observer versus matched stock `-ngl 10`: prompt TPS **-5.98%**, decode TPS **-4.75%**, end-to-end time **+5.49%**.
- C4 versus observer/cache-off: prompt TPS **-3.34%**, decode TPS **+1.98%**, end-to-end time **+1.38%**. The one-layer effect is small and mixed.
- C5 versus observer/cache-off: prompt TPS **+3.35%**, decode TPS **+9.75%**, end-to-end time **-5.77%**.
- Best ExpertFlow result (C5) versus strongest stock no-OOM: prompt TPS **-75.94%**, decode TPS **-71.88%**, end-to-end time **+287.33%**.
- C5 miss reduction versus C4: **57.33%**.
- C5 aggregate blocking-wall reduction versus C4: **45.45%**. The miss reduction therefore translated into materially lower measured blocking time.

## Exactness

- C4 and C5 exactly matched the canonical observer/cache-off prompt tokens and generated tokens for all three prompts and all three repetitions.
- C4 and C5 exactly matched all ordered observer router events, selected expert IDs, phases, token indices, and layer IDs.
- Each mode was internally deterministic across its three repetitions.
- The clean matched-stock build produced different generated tokens from the observer lineage on general and code, while translation matched. Full-offload stock differed on general, matched the observer on code, and matched both on translation. This cross-build/offload difference is reported, not hidden; it does not represent cache-on/cache-off drift.
- All processes exited; 60/60 performance and measurement records exist; no probe remained; settled system GPU use returned to 1,748 MiB after the matrix versus 1,729 MiB before it, within active-desktop variance.

## Multi-layer gate

| Gate | Result |
|---|---|
| Exact cache-off/cache-on tokens and routing | PASS |
| C5 lowers measured layer-24 blocking cost | PASS: -45.45% aggregate, -67.95% decode-phase blocking/token |
| No severe unexplained bookkeeping TPS regression | PASS: C5 decode TPS +9.75% versus observer |
| Stable memory and cleanup | PASS |
| Plausible broader-layer path | PASS, but required: one layer remains far behind full-offload stock |

Proceed only with the approved staged allocation ramp: two layers at 32 slots, then five, then all intended MoE layers, measuring exact parity, real arena/peak VRAM, per-layer hits/misses, blocking time, decode TPS, reserve, and KV/state headroom at every point. Do not begin with 64 slots across all layers.

Raw artifacts, the append-only ledger, scan evidence, manifest, 60 run directories, and machine summary are under `C:\models\expertflow\runs\performance-first-integration`.
