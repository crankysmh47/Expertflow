# Gate 3 Cross-Runtime Divergence Audit

Status: **CROSS-RUNTIME DRIFT EXPLAINED; GATE 3 STILL FAIL-STOP**

Recorded: 2026-07-15 PKT

Pinned source: `a7312ae94f801fc9c6786dc56e38df57b964f697`

Default/live state: `live_cache_enabled=false`

This bounded audit separates two questions that must not be conflated:

1. The earlier reference-versus-clean difference is expected build-dependent floating-point drift at a near-tied router boundary.
2. The clean CUDA runtime is not internally trace-safe across the representative prompt matrix. Code and translation change generated tokens when the router callback is enabled. That non-waived Gate 3 requirement still fails, so no cache work or one-layer recommendation is authorized.

No llama.cpp source or cache path was changed. The only repository source addition is opt-in tensor capture in the separate ExpertFlow diagnostic probe; its default logical output was regression-checked against the original probe.

## Complete run equivalence

The comparison manifest records the same 14,439,361,440-byte GGUF at SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`, the same 170 prompt bytes at SHA-256 `488245cb6b35fdeaba15be23b4b4adb00165dac1ed457a19f51f58e77f0573cc`, and identical prompt token IDs. Neither run applies the GGUF chat template. Both use greedy sampling, 64 generated tokens, 12 threads, one-token batch/ubatch, ten GPU layers, layer split on device 0, automatic flash attention, F16 K/V, mmap enabled, mlock disabled, and an effective padded context of 256.

The relevant build differences are explicit:

- The official b10002 core was produced in a VS 2026 environment with an LLVM vcpkg toolchain; the CUDA plugin used VS 2022, CUDA 12.4.131, and the official multi-architecture configuration.
- The clean build uses VS 2022 MSVC 19.39/v143, CUDA 12.8.93, Release `/O2 /Ob2 /DNDEBUG`, and only `120a-real`.
- The official runtime enables dynamic backends and CPU variants and selects `ggml-cpu-zen4.dll`; the clean runtime uses its single local `ggml-cpu.dll` with dynamic backends and all CPU variants disabled.
- The sibling CUDA/runtime DLLs and their hashes differ. Both execute on the same RTX 5060 Ti and driver 591.86.

These differences can change floating-point operation ordering and code generation even though model, prompt, tokenizer, decode, and runtime options are equivalent.

## First selected-set divergence

The prior report called this “prompt token 2.” That was a coordinate error: `2` is the BOS **token ID**. The actual location is event 24, forward 0, **token index 0**, token ID 2, layer 24.

| Measurement | Reference | Clean |
| --- | ---: | ---: |
| Rank 8 expert / probability | 113 / 0.0142387599 | 117 / 0.0141336974 |
| Rank 9 expert / probability | 117 / 0.0139993606 | 113 / 0.0140901292 |
| Rank 8/9 gap | 0.0002393993 | 0.0000435682 |

The clean top eight gains expert 117 and loses expert 113. Across all 128 probabilities, maximum absolute cross-runtime drift is `0.0004364327`, maximum relative drift is `0.0343542747`, and RMS drift is `0.000107767877`. The maximum drift exceeds both rank-8/rank-9 margins. Host ranking of the captured values exactly reproduces each runtime's selected set.

The raw router logits show the same boundary flip. Their rank-8/rank-9 gaps are `0.016955972` and `0.003087461`; maximum absolute logit drift is `0.044186026` and RMS drift is `0.0174278`.

## Bounded backward localization

- `inp_scaled` at token index 0 is byte-identical across runtimes: all 2,816 values and SHA-256 `dd13e66a7450af2e1df7f63f1380f77a2d30543dc4232fba6456154d04e3dd11` match.
- `l_out-0` is the first checked tensor that differs: maximum absolute drift `1.9000000009e-6`, RMS `9.362639294e-8`.
- By `attn_out-24`, all 2,816 values differ, with maximum absolute drift `0.33303225` and RMS `0.08687306`.

The runtimes therefore begin with identical model input. Small numerical differences start during layer 0 computation and accumulate before they flip the close layer-24 routing boundary. There is no evidence of tokenizer mismatch, malformed routing tensors, missing layers, or an argsort defect.

An initial `ffn_moe_argsort-24` capture was rejected as an explanatory metric because it is the full I32 argsort permutation, not router scores. This failed approach remains in the ledger.

## Controlled local build comparison

The existing exact-source MSVC Debug CPU build and the clean Release CPU path were run with the same prompt and router capture. Both selected the same top eight at token index 0/layer 24, but their 128 probabilities still differed: maximum absolute drift `0.0002273954`, RMS `0.00003099388464`. Release took 8.664 seconds and Debug took 58.492 seconds. This controlled result independently supports build-dependent numerical variation; it is not used as a performance comparison.

## Clean-runtime stability matrix

Each domain has three trace-off and three trace-on deterministic runs with 16 generated tokens. Each mode is internally token-stable; all traced repeats have identical ordered router selections, complete 30-layer forwards, eight experts per event, and causal hook order. No llama/probe process persists after the matrix.

| Domain | Trace-off repeats | Trace-on repeats | Off/on token parity | Ordered router repeats | Settled GPU range |
| --- | --- | --- | --- | --- | ---: |
| General chat | PASS | PASS | PASS | PASS | 23 MiB |
| Code | PASS | PASS | **FAIL** | PASS | 37 MiB |
| Translation | PASS | PASS | **FAIL** | PASS | 29 MiB |

The code prompt changes from generated token 0 when tracing is enabled; the translation prompt diverges at generated token 2. These are true token-array differences, not timestamp or metadata differences.

The pinned scheduler explains the perturbation boundary. Without an evaluation callback it computes a whole backend split asynchronously. With the callback installed, it computes synchronized graph views ending at each requested router tensor. Thus the current CUDA tracing mechanism is observationally intrusive. Determinism within each path does not satisfy trace-off/on parity.

## Decision

The old reference-versus-clean difference is classified **benign build-dependent near-tie drift** and is not, by itself, evidence of corrupted inference. However, the user's decision rule also requires the clean build to be internally trace-safe. It is not across the representative matrix.

Therefore:

- Gate 3 remains **FAIL-STOP** on clean trace-off/on token parity.
- The clean Release binary is not promoted to the cache correctness baseline yet.
- No one-layer blocking-slot recommendation is issued.
- No cache source, slot allocation, transfer path, predictor, async stream, or MTP work begins.
- `live_cache_enabled=false` remains mandatory.
- The protected Observatory and pinned llama.cpp checkout remain untouched.

## Evidence

External root: `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit`

- `audit-summary.json`: machine-readable verdict and artifact inventory.
- `comparison-manifest.json`: complete option/build comparison.
- `score-comparison.json` and `logit-comparison.json`: all 128 router values and boundary metrics.
- `inp-scaled-comparison.json`, `l-out-0-comparison.json`, and `attn-out-24-comparison.json`: bounded drift localization.
- `stability-summary.json`: per-prompt, per-run tokens, traces, timings, host/GPU memory, and strict validation.
- `alternate-build/release-vs-debug-router.json`: controlled local Release/Debug result.
- `commands.jsonl`: append-only commands, failures, corrections, measurements, and decisions.

Final verification passed 87 tests, the 8-event/64-demand judge replay, diagnostic probe build/contract checks, JSON/JSONL parsing, recorded artifact hashes, `git diff --check`, protected/source identity and cleanliness, and no-cache environment/process checks. A fresh pre-commit rerun passed 87 tests in 0.43 seconds and reproduced the fixture again. `audit-summary.json` hashes to `142f579a466a80b377ded5a99fcb5362d450c87bd1dbc123cf176664cda60cdf`; `verification.json` hashes to `f94314f3967d5cc829105bcd85b4290b96b1145a94c3ec3a18c619198a4226f2`; the 39-record pre-commit ledger hashes to `ec353b1526f9f97853222bda8add2b3b58864b116f498bd13db3249dc71d511d`.
