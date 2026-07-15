# Gemma 4 Q4 expert-size and transfer evidence

Date: 2026-07-15 PKT

- **Model:** `google/gemma-4-26B-A4B-it-qat-q4_0-gguf@21bfe2a8c89118c9a1a2aa242934fc4d1c0fff15`
- **Model SHA-256:** `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- **Runtime source:** `ggml-org/llama.cpp@a7312ae94f801fc9c6786dc56e38df57b964f697` (`b10002`)
- **GPU:** NVIDIA GeForce RTX 5060 Ti, 16,311 MiB reported VRAM
- **Expert-size gate:** PASS
- **First transfer-curve gate:** PASS
- **Live-runtime gate:** CONDITIONAL / OBSERVATORY-FIRST

## Encoded expert size

The pinned revision's `gguf-py` reader memory-mapped the GGUF and read tensor metadata without copying weight arrays. All 30 MoE layers have the same three expert tensors and 128 contiguous expert slices:

| Per-layer tensor | GGUF type and shape | Layer bytes | Bytes per expert |
| --- | --- | ---: | ---: |
| `ffn_down_exps.weight` | Q4_0 `[704, 2816, 128]` | 142,737,408 | 1,115,136 |
| `ffn_gate_up_exps.weight` | Q4_0 `[2816, 1408, 128]` | 285,474,816 | 2,230,272 |
| `ffn_down_exps.scale` | F32 `[128]` | 512 | 4 |
| **Encoded expert total** | — | — | **3,345,412 bytes / 3.190434 MiB** |

The scale array for all experts and layers is only 15,360 bytes and can remain resident; a cold expert's two weight slices total 3,345,408 bytes.

The CUDA backend does not repack these tensors while loading them. Its `set_tensor` path copies the supplied byte range with `cudaMemcpyAsync`. Its allocation-size path returns `ggml_nbytes(tensor)` plus end padding for quantized matrix rows, and CUDA tensor starts are 128-byte aligned. If a future cache represents each expert as three independent backend tensors, a conservative source-derived slot projection is 3,346,048 bytes / 3.191040 MiB after Q4 row padding and 128-byte alignment. This is a projected cache layout, not a measurement of a cache that does not exist yet.

At that projected size:

- eight slots in each of all 30 MoE layers require 765.85 MiB;
- eight slots in the 21 layers left on CPU by the current 10-layer offload require 536.09 MiB;
- the measured 7,234 MiB configurable headroom can hold 2,266 such slots before any additional cache-specific staging or fragmentation reserve.

The capacity arithmetic establishes feasibility, not an allocation recommendation.

## Transfer command and artifact

```powershell
uv run expertflow transfer-benchmark `
  --cudart C:\models\expertflow\dependencies\llama-b10002\runtime\cudart64_12.dll `
  --payload-bytes 1115136 `
  --payload-bytes 2230272 `
  --payload-bytes 3345412 `
  --payload-bytes 26763296 `
  --batches 30 `
  --copies-per-batch 50 `
  --warmup-copies 10 `
  --device 0 `
  --output C:\models\expertflow\runs\transfer-q4\transfer.json
```

The command uses only Python's standard library and the pinned CUDA runtime API. It records 30 per-copy batch averages after 10 warm-up copies. Device timing uses CUDA events; host staging and host-visible duration use a monotonic wall clock.

- CUDA runtime DLL: 553,984 bytes, SHA-256 `d28e42265da7462162a54da6b7a99ea4fa2caf8139d862bb500db875d0b32dfc`
- CUDA runtime version: 12.4 (`12040`)
- CUDA driver API version reported by the runtime: `13010`
- Result: 23,265 bytes, SHA-256 `c2bb5a820e99552cb161aa37216725bb806934637a7deb4a77083b0733539962`

## Measured curve

| Payload | Bytes | Pageable→pinned mean / p95 | Pageable→GPU event mean / p95 | Pinned→GPU event mean / p95 |
| --- | ---: | ---: | ---: | ---: |
| Down slice | 1,115,136 | 0.0152 / 0.0158 ms | 0.1371 / 0.1414 ms | 0.0788 / 0.0789 ms |
| Gate/up slice | 2,230,272 | 0.0886 / 0.0913 ms | 0.2502 / 0.2552 ms | 0.1563 / 0.1564 ms |
| Combined encoded expert | 3,345,412 | 0.1289 / 0.1305 ms | 0.3568 / 0.3643 ms | 0.2337 / 0.2339 ms |
| Eight-expert layer fill | 26,763,296 | 1.1673 / 1.3877 ms | 2.8343 / 2.8950 ms | 1.8620 / 1.8636 ms |

The pinned→GPU means sustain 13.18–13.39 GiB/s across the measured sizes. The two actual weight-slice transfers for one pinned expert sum to 0.2350 ms mean. Staging those two slices from already-warm pageable memory adds 0.1038 ms, for a serialized two-leg mean of 0.3388 ms. The separately measured combined payload agrees closely with the two pinned transfers.

These are repeated warm-buffer, default-stream microbenchmarks with no competing model compute, disk I/O, page faults, scheduler work, tensor bookkeeping, or cache eviction. They are a transfer lower bound for the tested machine, not predicted or measured end-to-end token savings.

## Pooled single-copy physical-feasibility checkpoint

The later bounded feasibility stage repeated the benchmark in three independent processes and added idle-stream single-copy CUDA events plus host API-call timing. Each payload/memory point pools 600 single-copy samples. Thirty batches of 50 copies per trial remain the sustained-bandwidth measurement.

- Aggregate: `C:\models\expertflow\runs\transfer-q4-physical\aggregate.json`
- Bytes: 495,063
- SHA-256: `fb90c8820085f80849977cbf7849de2c899d9c1a4dfd20b5e3fe20e63244b94b`
- Trials: `a4c522cc75cd094fee9697fb3ce76ca9b3dffb2947dd3dac033d4087c8a3391c`, `7c6d825f51de09e92c29db18d54e7eb9dabe3d4117187d4cc3426386d23bf1ae`, `ffd526a97b7f3cd8b4bd829bb3605a82d15f2970fef1a3c6a566973c708ae9f5`
- Environment: RTX 5060 Ti, WDDM, CUDA Runtime 12.4, driver API 13.1, default stream, 0% GPU compute utilization before launch, no model process

| Payload | Pinned H2D single-copy p50 / p95 | Pinned sustained bandwidth | Pageable H2D p50 / p95 | Pageable-to-pinned p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Down slice, 1,115,136 B | 0.079200 / 0.079456 ms | 13.21 GiB/s | 0.080096 / 0.080416 ms | 0.015236 / 0.015708 ms |
| Gate/up slice, 2,230,272 B | 0.156608 / 0.156832 ms | 13.31 GiB/s | 0.164288 / 0.165024 ms | 0.086979 / 0.088304 ms |
| Encoded expert, 3,345,412 B | 0.233984 / 0.234208 ms | 13.35 GiB/s | 0.264784 / 0.279680 ms | 0.129595 / 0.133816 ms |
| Aligned slot, 3,346,048 B | 0.234016 / 0.234272 ms | 13.35 GiB/s | 0.264672 / 0.276704 ms | 0.129780 / 0.132172 ms |
| Eight aligned slots, 26,768,384 B | 1.859840 / 1.860128 ms | 13.41 GiB/s | 2.599504 / 2.666592 ms | 1.056571 / 1.083898 ms |

A one-byte pinned copy measures the host `cudaMemcpyAsync` API call at approximately 0.0013 ms p50 and 0.0015 ms p95. This is copy enqueue overhead on the tested idle stream, not CUDA kernel-launch overhead. For the two real weight slices, pinned event latency sums to 0.235808 ms p50 and 0.236288 ms p95; their host API calls sum to roughly 0.0026 ms p50 and 0.0035 ms p95.

This pooled result supersedes the earlier batch-average transfer value as the simulator timing input. It does not change the lower-bound limitation: there is still no concurrent model compute, cache bookkeeping, eviction, or WDDM contention in the microbenchmark.

## Gate decision

The 24-hour empirical requirements for real Gemma expert sizes and a first measured transfer curve now pass. The live-runtime gate does not. A single global cache over the parity-safe stratified prefill trace reaches 33.13% static-8 and 35.13% LRU-8 on the 21 CPU-resident target layers. Static reaches 92.77% at 64 slots/layer, but that hotset is fit and evaluated on the same small workload. ExpertFlow has not measured CUDA per-layer compute deadlines or demonstrated that a transfer issued by a real cache arrives before expert use. It also has no same-runtime reactive-versus-cached end-to-end comparison.

Continue with the Observatory and use these values in the offline deadline simulator. The [stratified capacity curve](q4-capacity-curve.md) now consumes the measured slot and transfer values, but `live_cache_enabled=false` remains mandatory until deadline feasibility and exact end-to-end behavior pass.
