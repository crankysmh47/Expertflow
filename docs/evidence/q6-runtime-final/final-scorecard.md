# Final scorecard

| Requirement | Result |
|---|---|
| Truthful pristine Q6 baseline | PASS: 22.967 mean decode TPS, 512 tokens × 3 cold processes |
| 256 MiB baseline margin | PASS: 3,098.656 MiB peak process VRAM |
| Patched-disabled equivalence | PASS: exact tokens and routing |
| CPU-authoritative, CUDA-executed narrow Q4 island | PHYSICAL PASS: 408.375 MiB complete layer-0 shadow |
| Exact layer-0 island output | FAIL: generated token 1 differs |
| Exact router IDs/order | FAIL: first order change at prefill token 0, layer 13 |
| Stable deterministic candidate | PASS: three identical enabled repetitions |
| Q6 reactive product | NOT RUN: blocked by Stage 1 |
| Primary 1.25× target (28.708 TPS) | NOT EVALUATED |
| Sanitizer/final 512-token Q6 product run | NOT RUN: exactness stop precedes it |

No ExpertFlow Q6 throughput or speedup claim is made.

NARROW PLACEMENT STOP
