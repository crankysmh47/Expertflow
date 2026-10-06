# Live product benchmark summary

## Continuous batching

The final five-repetition, four-slot server profile completed 20 of 20 requests. ExpertFlow delivered **35.6699 aggregate generated TPS** versus **24.5231 stock**, a **45.4546%** increase under the metric `total generated tokens / request-batch wall time`. Peak process-owned VRAM was **11,996.8125 MiB**, leaving **4,314.1875 MiB** on the 16,311 MiB device. Median request latency was 13.314 seconds, p95 was 20.311 seconds, and median/p95 TTFT were 2.517/7.984 seconds.

This is a live continuous-batching result, not the 28.13 TPS single-request `llama-cli` result. Sample variance was high (58.179 TPS squared), and deterministic concurrent response hashes were not reproduced across all runs. The release reports that limitation directly.

## Context frontier

Both modes allocated and ran 8K, 16K, 32K, 64K, 128K, 256K, and 512K contexts. ExpertFlow also passed 786,432, 917,504, and 950,272 allocations, then failed at 983,040 and 1,048,576 during static-shadow CUDA allocation. Stock passed 1,048,576.

Each passing point processed 385 prompt tokens and generated 32 tokens. Therefore the allocation is measured, but the full context capacity was not filled. The shippable context profile is conservatively capped at Gemma's 262,144-token training context: 30.2382 decode TPS, 15,635.582 MiB peak, and 675.418 MiB reserve. This is a bounded live allocation proof, not a maximum filled-context claim.
