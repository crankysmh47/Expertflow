# Benchmarking ExpertFlow

The headline comparison uses `google_gemma-4-26B-A4B-it-Q6_K.gguf` (22,862,575,520 bytes, SHA-256 `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`) on a Windows 11 x64 machine with an NVIDIA RTX 5060 Ti 16 GB, driver 591.86, CUDA 12.8.93, and MSVC v143 14.39.33519.

The runtime is llama.cpp `451224ab4d12a616dc3e16e8c8063f4b331f531c`, based on upstream `a7312ae94f801fc9c6786dc56e38df57b964f697`. Stock and ExpertFlow used the same binary, Q6 model, prompt, `-ngl 99`, `--cpu-moe`, 12 threads, seed 42, temperature 0, 2,048-token context, and CUDA graph mode. Batch and ubatch were not overridden for the generation comparison, so the pinned runtime defaults applied equally to both modes.

Stock ran without ExpertFlow environment variables. ExpertFlow added:

```text
LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER=0,1,2,3,4,5,6,7,8,9,15,20
LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE=1
```

Ten matched cold-process pairs generated 512 decode tokens per run. Decode TPS is generated tokens divided by the runtime's measured decode duration. Process-owned peak VRAM came from `nvidia-smi` PID sampling, not global device allocation. The strongest fair stock reference was 22.967 TPS; ExpertFlow averaged 28.13 TPS, a 22.48% improvement against that reference.

The four-slot result used one loaded server, four concurrent requests, 512 generated tokens per repetition, and five cold-server repetitions. Its 35.6699 TPS is aggregate throughput and must not be compared with a single-stream number as if the protocols were identical.

The 262,144-token context result is an allocation test. It processed 385 prompt tokens and 32 generated tokens, 417 total. It is not evidence that a 262,144-token prompt was filled and evaluated.

Public Q4, MTP, cloud, prompt-processing, different-backend, and aggregate-throughput results answer different questions. They are useful references, but a direct speed comparison requires the same model quantization, hardware, runtime, workload, token count, concurrency, CUDA settings, and placement policy.

Machine-readable values live in `release/expertflow-build-week/evidence/release-scorecard.json`.
