# Q6 selected-layer static performance experiment

## Verdict

`Q6 PERFORMANCE STOP`

The four Q6 static CUDA islands produced a statistically positive matched improvement, but not a product result. The authoritative graph-disabled run moved decode throughput from **19.95 to 21.44 TPS**, a paired mean improvement of **7.53%** with a **[4.66%, 10.42%]** matched-bootstrap 95% interval. Quality improved and stability passed. However, 21.44 TPS remains **6.65% below** the frozen strongest fair-stock result of 22.967 TPS and **33.90% below** the 28.709 TPS product target.

Conditional pass is rejected because no quantified credible path reaches the product target. The measured peak and 256 MiB margin permit at most 15 more full Q6 layer bundles. Even the deliberately optimistic assumption that every added layer equals the average benefit of the selected four projects only 27.03 TPS, and Q1b verified no additional low-sensitivity layer set. Reactive caching, reduced capacity, prediction, new layer selection, scheduler work, and kernel work therefore remain closed.

## Stage 1: measured bottleneck

The stock-equivalent `-ngl 99 --cpu-moe` placement was audited rather than inferred from the reported 31/31 CUDA layers:

- Router/top-k: CUDA.
- Authoritative routed-expert tensors: CPU-mapped.
- Selected `MUL_MAT_ID` and activation path: CPU.
- Result boundary: copied back into the CUDA graph.

The synchronized diagnostic split profiler attributes 1,234,942 of 13,084,288 microseconds (9.44%) to the four selected expert splits, above the predeclared 5% physical-plausibility threshold. This timing is diagnostic only because profiling deliberately synchronizes each split.

| Layer | Split | CPU expert split time | Diagnostic share |
|---:|---:|---:|---:|
| 0 | 2 | 491,818 us | 3.76% |
| 1 | 4 | 361,790 us | 2.77% |
| 15 | 32 | 211,817 us | 1.62% |
| 20 | 42 | 169,517 us | 1.30% |

## Stage 2: static placement proof

The existing Q1b persistent-island implementation accepted Q6 without source changes. Each selected layer preloaded all 128 experts once with identity mapping:

| Component | Type and shape | Bytes/layer |
|---|---|---:|
| Gate/up | Q6_K `[2816,1408,128]` | 416,317,440 |
| Down | Q8_0 `[704,2816,128]` | 269,615,104 |
| Down scale | F32 `[128]` | 512 |
| Complete bundle | 5,358,852 bytes/expert | 685,933,056 |

The exact four-layer shadow allocation is **2,743,732,224 bytes**. CPU sources remained authoritative; no hidden full routed-expert CUDA duplicate, per-token allocation, repacking, NaN, CUDA error, corruption, progressive growth, or teardown failure was observed. The feature remains environment-gated and disabled by default.

## Stage 3: matched performance

Ten cold-process pairs generated 512 tokens each with identical Q6 GGUF, prompt, context 2048, seed 42, temperature 0, 12 CPU threads, `-ngl 99 --cpu-moe`, and `GGML_CUDA_DISABLE_GRAPHS=1`. Order alternated OFF/ON and ON/OFF. All 20 ordinary runs are included; each mode produced one deterministic response hash across its ten repetitions.

| Metric | OFF | ON |
|---|---:|---:|
| Decode TPS, mean | 19.95 | 21.44 |
| Decode TPS, median | 20.25 | 21.40 |
| Decode TPS, sample SD | 0.595 | 0.882 |
| Prompt TPS, mean | 19.33 | 20.96 |
| Cold-process wall time, mean | 34.57 s | 35.21 s |
| Process-owned peak VRAM | 3,094.656 MiB | 5,720.762 MiB |

Paired improvement values were `8.82, 13.73, 10.47, 2.45, 14.81, 3.90, 0.50, 3.03, 10.26, 7.35%`. Their mean is 7.53%, median 8.09%, sample SD 4.92 percentage points, and 100,000-resample matched-bootstrap interval `[4.66%, 10.42%]` with seed 20260719.

A separate one-pair llama-server streaming diagnostic measured OFF/ON TTFT of 281.45/342.41 ms, p95 token latency of 54.61/52.73 ms, and p99 of 78.89/69.26 ms. These interface-specific latency values are not mixed with the authoritative CLI throughput statistics.

The first complete 10-pair batch used the build-default graph setting. It was quarantined as a whole before the authoritative verdict because the approved static protocol required CUDA graphs disabled. A latency-server startup failure (`0xC0000135`, missing child DLL search paths) was preserved, fixed test-first in the harness, and rerun successfully. No ordinary slow performance run was removed.

## Stage 4: held-out quality

The independently frozen Q1b WikiText-103 validation corpus produced all 8,184 paired NLL records. Bootstrap blocks were 128 tokens within eight 2,048-token chunks; 10,000 resamples used seed 20260718.

| Metric | OFF | ON |
|---|---:|---:|
| Mean NLL | 9.686361 | 9.612071 |
| Perplexity | 16,096.566 | 14,944.088 |
| Relative change | - | **-7.16%** |

The paired 95% interval is `[-11.45%, -2.59%]`; its upper bound is below the +1% gate. The frozen 100-item MMLU subset scored **49/100 OFF and 52/100 ON**. All 15 changed answers were repeated in ON mode and matched item identity, prediction, token, and content exactly. Both quality gates pass.

## Reproduction boundary

- ExpertFlow branch: `codex/q6-selected-static`, based on Q1b `834cf3d`.
- llama.cpp branch: `codex/q6-selected-static-llama`, profiler commit `d8e387fc`, based on Q1b `38dce264` and upstream `a7312ae94f801fc9c6786dc56e38df57b964f697`.
- Build: Release, Ninja, MSVC 19.39/v143, CUDA 12.8.93, `GGML_CUDA=ON`.
- GPU/driver: NVIDIA GeForce RTX 5060 Ti 16 GB, driver 591.86.
- Model SHA-256: `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.

Detailed commands and quarantined attempts are in `command-ledger.md`. Raw large logs remain outside Git at `C:\models\expertflow\runs\q6-selected-static`; every verdict-bearing raw artifact is hash-identified in the JSON evidence.

## Exact next recommendation

Freeze this isolated Q6 static branch and do not begin reactive caching or another runtime architecture experiment. Return to the protected Observatory/release submission floor and present this as honest bounded negative product evidence: the static islands relieve a real measured CPU bottleneck and preserve quality, but do not beat the strongest stock runtime or provide a memory-feasible path to the required 28.709 TPS.
