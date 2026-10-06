# Final sprint Stage 0 feasibility

## Conclusion: STOP

Do not begin R1. Stock reached **917,504 tokens** at `-ngl 10` with a measured process-owned peak of **15,598.27 MiB**, leaving the required 256 MiB margin on the 16,311 MiB GPU. The required 1.5× target is therefore at least **1,376,256 tokens**.

The whole-model cache moves in the wrong direction for this capability. A 112-slot arena is **10,719.87 MiB** and a 96-slot arena is **9,188.46 MiB**. After removing the nine full CUDA expert layers represented in the measured `-ngl 10` fixed footprint, adding the whole-model arena, retaining baseline scheduler/workspace costs, and reserving 256 MiB, the conservative 13 KiB/token fit projects only about **312K** tokens for 112 slots and **432K** for 96 slots. Both are below stock's measured 917,504-token lower bound, not 1.5× above it. No predefined smaller-GPU fallback exists in the repository or sprint specification.

## Reproduction package

- Upstream: `a7312ae94f801fc9c6786dc56e38df57b964f697`
- Final runtime: `9039ea21e493619da5991dfb699a94a60bb3a28c`
- Exact source archive: `llama-cpp-9039ea21-source.zip`, 37,207,260 bytes, SHA-256 `4c6bf0d09bb178f5b7d0234de08a309437cd6afde6ad2139a9fec5a152300144`
- Exact `ggml-backend.cpp` archive: SHA-256 `369704fa06c69d1a7e6e5494320b71d38096f61dc06ae8a31905452d7a036bda`; Git blob `24719e608a46c0406281ec697457076f84e86aba`
- Twelve ordered `format-patch` files and the archives are under `C:\models\expertflow\runs\final-sprint-stage0\runtime-package`.
- Build: Release/Ninja, MSVC 19.39.33523, CUDA 12.8.93, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120a-real`; driver 591.86.
- Model: Gemma 4 26B A4B Q4_0, 14,439,361,440 bytes, SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`.

The machine-readable manifest contains binary hashes, patch identities, exactness-reference hashes, raw-result paths, and all calculations.

## Portability and exactness freeze

Private worktree constants were removed from source-contract tests. A clean checkout runs **172 passed, 2 skipped** without external runtime source. Setting `EXPERTFLOW_LLAMA_SOURCE` to the patched fork runs the skipped modules and produces **8 passed**. The canonical cache-disabled observer reference is frozen by token and router-trace hashes for general, code, and translation prompts in `stage0-results.json`.

## Expert-bundle and remap audit

The GGUF inventory covers all 30 routed layers and all 3,840 expert objects. Every object contains exactly:

| Component | Type | Bytes/expert |
|---|---|---:|
| fused gate/up | Q4_0 | 2,230,272 |
| down | Q4_0 | 1,115,136 |
| down scale | F32 | 4 |

The packed bundle is 3,345,412 bytes. The runtime identifies, allocates, bounds-checks, and transfers these same three components. There is one authoritative remap write to the shared selected-ID tensor after all three component transfers and committed-mapping validation; both `MUL_MAT_ID` consumers and the scale lookup therefore use the same physical IDs. The current enabled path keeps the corresponding sources host-backed, so no full CUDA expert duplicate was found. Whole-model duplicate freedom remains a projection because R1 was not implemented.

## Measured memory slope and stock frontier

Process-owned Windows GPU dedicated-memory counters were sampled every 200 ms and cross-checked against NVIDIA system totals. On the stable `-ngl 10` series from 2,048 through 262,144 context, OLS gives **11.875 KiB/token**, upper 95% slope **12.639 KiB/token**, `R²=0.99486`; budgeting uses **13 KiB/token**.

| Context | Fastest tested stock `-ngl` | Decode TPS | Process peak MiB |
|---:|---:|---:|---:|
| 2,048 | 99 | 97.3 | 13,886.70 |
| 8,192 | 99 | 97.8 | 13,886.70 |
| 16,384 | 99 | 95.8 | 13,886.70 |
| 32,768 | 20 | 40.8 | 10,276.77 |
| 65,536 | 20 | 41.3 | 10,282.68 |
| 98,304 | 20 | 37.4 | 11,348.91 |
| 131,072 | 20 | 41.0 | 12,020.97 |
| 917,504 | 10 | 13.9 | 15,598.27 |

The 917,504-token run is a stable allocation/decode feasibility point with a short prompt and eight generated tokens, not a long-context quality or fully populated-attention throughput claim.

## Arena accounting

Thirty full expert bundles occupy 12,846,382,080 bytes. Consolidated per-component arena arithmetic, including 512 bytes of observed alignment per layer, gives:

| Slots/layer | Arena MiB | Gross expert savings MiB | Conservative projected max context |
|---:|---:|---:|---:|
| 112 | 10,719.87 | 1,531.39 | ~312K |
| 96 | 9,188.46 | 3,062.80 | ~432K |

Device staging is zero because prediction is disabled. Mapping/LRU state is host-resident. Existing scheduler and compute buffers are retained in the measured fixed-footprint estimate. The 15,360-byte arena alignment overhead is included. A hidden full expert duplicate would only worsen the result; none is credited as savings before R1 proof.

The declared feasibility gate fails, so Stage 0 stops here.
