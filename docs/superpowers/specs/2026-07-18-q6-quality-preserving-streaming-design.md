# Q6 Quality-Preserving Expert Streaming Design

## Decision

ExpertFlow will pursue quality-preserving Q6 expert streaming. The cache remains exact at the weight, mapping, transfer, and routing-input level, but CPU-versus-CUDA floating-point output is not required to be bit-identical. Every divergence remains visible in the evidence.

The protected `codex/q6-runtime-final` branch at `378ce57` remains the terminal exactness result and rollback floor. This work proceeds only on `codex/q6-quality-preserving`. ExpertFlow Deploy remains the protected fallback; it is not the primary product while runtime gates pass.

## Alternatives considered

1. **Deterministic benchmark-first quality contract — selected.** Freeze established datasets and machine-scored thresholds before further cache work. This prevents favorable prompt selection and gives the runtime a defensible quality claim.
2. **GPT-5.6 semantic judging.** Useful for qualitative review, but not reproducible enough to decide a hard gate. It may annotate samples after machine metrics are frozen.
3. **Build reactive caching before measuring quality.** Faster initially, but creates selection bias and risks spending the remaining schedule on an already-disqualified numerical path. Rejected.

CUDA VMM is not the automatic fallback. It may receive a separate three-to-four-hour microproof only when quality is the sole failing dimension and exactness becomes necessary. Cache instability or inadequate performance switches directly to ExpertFlow Deploy.

## Frozen quality protocol

The comparison is always same model, prompt/input tokens, context, seed, sampler, batch, microbatch, threads, build, and output length. The only changed variable is the quality-preserving runtime feature.

Before the first scored run, a manifest freezes immutable dataset revisions, source hashes, item IDs, ordering, prompts, answers, executable hashes, model hash, command lines, and runtime environment. The manifest cannot be regenerated after candidate results are inspected.

### Perplexity

- Dataset: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, test split.
- Evaluation window: the first four complete 2,048-token chunks produced by the pinned Gemma tokenizer after joining non-empty test rows with newlines.
- Tool: the matched `llama-perplexity` build.
- Gate: candidate perplexity may increase by at most **0.5% relative** to the feature-disabled baseline.

### Fixed task score

- Dataset: `cais/mmlu`, test split.
- Sample: ten fixed subjects with ten questions each, selected by SHA-256 ordering using salt `expertflow-option1-v1`; 100 items total.
- Subjects: abstract algebra, college computer science, college mathematics, computer security, conceptual physics, electrical engineering, high-school world history, machine learning, moral scenarios, and professional law.
- Prompting: one frozen zero-shot multiple-choice template; greedy decoding; answer extraction accepts only the first standalone `A`, `B`, `C`, or `D` after the answer marker.
- Gate: candidate accuracy may decline by at most **1.0 percentage point** from the feature-disabled baseline.

### Determinism, routing, and long generation

- Run the candidate three times in fresh processes. Prompt tokens, generated tokens, and router events must be identical across candidate repetitions.
- Report top-8 set overlap, ordered-position overlap, exact-set rate, and exact-order rate by layer and phase. These are descriptive diagnostics, not substitutes for the perplexity and task gates.
- Run six frozen multi-domain prompts for 256 generated tokens each: general explanation, code, math/reasoning, translation, structured output, and deliberate topic shift.
- Candidate outputs must be non-empty, valid UTF-8, free of NaN/Inf runtime diagnostics, and complete without CUDA or allocation errors.
- Across the six prompts, candidate repeated-4-gram rate may not exceed baseline by more than five percentage points, and distinct-2 may not fall more than five percentage points.

## Runtime progression

### Q1 - Static Q4 fidelity

Restore the bounded layer-0, 128-expert, CPU-authoritative CUDA island in an isolated llama.cpp worktree. Use persistent complete gate/up, down, and scale shadows. Router/top-k and combine/residual remain on their original backend. The feature is disabled by default. Run the complete frozen quality protocol.

### Q2 - Small reactive Q4 cache

Only after Q1 passes, replace static residency for layer 0 with exact synchronous reactive loading. Test 96 then 64 slots; 32 is a stress-only configuration. Slot publication occurs only after the complete bundle is ready. Expand to at most layers 0-2 after layer 0 passes stability and quality.

### Q3 - Quantization-independent Q6 static proof

Derive bundle components, byte counts, strides, type traits, and alignment from GGML metadata. Run one Q6 static layer before eviction. Stop if existing CUDA kernels cannot consume the Q6 tensors or a replacement kernel is required.

### Q4 - Whole-model Q6 reactive product

Test capacities in order `88 -> 80 -> 72 -> 64`, choosing the largest that preserves a reproducible 256 MiB VRAM margin. Compare against the frozen fair stock baseline of **22.966667 decode TPS**. Reactive Q6 must reach at least 90% of stock before prediction work; final product success still requires at least **28.708333 decode TPS**.

### Q5 - Existing predictor and product shell

Reuse the frozen predictor only when reactive Q6 passes. Keep prediction only when blocked transfer time improves by at least 15% and decode TPS improves by at least 3% over reactive at the same memory budget. Package the successful path through `expertflow probe`, `expertflow run`, and `expertflow compare`.

## Stop rules

Stop runtime work and switch to ExpertFlow Deploy when any occurs:

1. Perplexity increases by more than 0.5%.
2. MMLU accuracy declines by more than 1.0 percentage point.
3. Candidate determinism, memory safety, or stable teardown fails.
4. Long-generation degeneration crosses either frozen threshold.
5. Reactive Q6 remains below 90% of fair stock.
6. Prediction is required merely to reach stock performance.
7. Another scheduler redesign, new GGML operation, or replacement CUDA kernel becomes necessary.
8. Runtime work threatens a complete submission and demo.

No quality, throughput, overlap, sanitizer, or product claim is made until its corresponding measurement passes. Detailed commands, durations, failures, hashes, and results remain append-only under `docs/evidence/q6-quality-preserving/`.
