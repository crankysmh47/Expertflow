# Performance-First Diagnostic Benchmark Design

## Scope

This milestone measures five existing, immutable runtime modes before any multi-layer llama.cpp edit:

1. strongest stable stock CUDA offload that does not OOM;
2. matched stock CUDA at `-ngl 10`;
3. canonical observer-enabled runtime with cache disabled;
4. exact layer-24 C4 eight-slot cache;
5. exact layer-24 C5 32-slot reactive LRU cache.

The benchmark uses the same Gemma 4 Q4 file, three fixed prompts (general, code, translation), greedy sampling, 64 requested decode tokens, context/batch settings, 12 CPU threads, and one warmup plus three measured repetitions. Each process starts from a fresh model load. The stock offload scan is diagnostic and precedes the fixed matrix.

## Measurement contract

An instrumented benchmark probe is built against each already-verified runtime without modifying llama.cpp. It writes one JSON record containing llama.cpp prompt/decode counters, host-wall process-phase measurements, per-decode host-wall latencies, token IDs, and trace/cache artifact paths. System GPU usage is sampled externally; it is labeled system-wide peak, not process-owned allocation.

`llama_perf_context()` supplies prompt-evaluation and decode totals and counts. Time to first token is host wall from the first prompt decode start through the first sample. Token p50/p95 is calculated only from already captured host-wall decode-step samples. Cache copy duration remains the cache event's blocking host-wall duration around copy and synchronization. It is never called CUDA-event latency or copy/compute overlap.

## Correctness and accounting

Prompt and generated token arrays must match the matched stock reference. Router parity is compared among observer, C4, and C5 full traces. Cache event accounting must reconcile demands, hits, misses, bytes, and aggregate blocking duration. The report includes repetition values, arithmetic mean, sample variance, explicit baseline deltas, C5 miss reduction versus C4, and whether reduced misses reduced measured blocking time.

No multi-layer implementation begins unless exactness passes, C5 lowers layer-24 blocking cost, bookkeeping avoids a severe unexplained TPS regression, memory is stable, and broader coverage remains plausibly beneficial.
