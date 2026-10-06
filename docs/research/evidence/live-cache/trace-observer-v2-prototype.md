# Trace Observer v2 One-Layer Prototype

**Verdict:** `FAIL-STOP — CONFIGURED BOUNDARY NOT REACHED`

The bounded prototype compiled successfully with MSVC 19.39 and CUDA 12.8 after modifying only `ggml/src/ggml-backend.cpp`. It registered no evaluation callback, requested no additional tensor, added no synchronization or graph view, and kept all capture storage fixed and preallocated. The llama.cpp prototype remains uncommitted because V3 did not pass.

## Validation result

The frozen configuration used layer 24, ten GPU layers, one-token batch/ubatch, 12 threads, greedy sampling, 16 generated tokens, and the same general, code, and translation prompts that exposed the callback failure. Every runtime invocation kept `--trace-mode disabled`.

| Stage | Runs | Result |
|---|---:|---|
| V0 compiled, observer unset | 9/9 | Exact token equality with the prior clean-runtime artifacts across all prompts and repetitions; deterministic; no observer artifact. |
| V1 observer `noop` | 9/9 | Exact V0 token parity; deterministic; zero records; no overflow/contract/canary failure. |
| V2 layer-24 metadata | 1/1 | Token parity held, but the observer recorded 0 of 53 expected forwards. Mandatory stop. |
| V3 selected IDs | 0 | Not started. |

V0 median durations were 6.8806 seconds general, 6.9201 code, and 6.8646 translation. V1 medians were 6.8501, 6.8569, and 6.8285 seconds respectively. These small negative differences are descriptive noise, not a performance claim. No V3 overhead exists because V3 was not run.

The V2 artifact is a valid 40-byte header: mode metadata, layer 24, capacity 4,096, count zero, overflow false, contract error false, and canaries valid. Its SHA-256 is `c299093240205bd5039db374a3afba918748cb0588936dbbab726e02fc74d8ba`.

## Interpretation

The source boundary itself exists, but the configured layer does not traverse that host-weight `MUL_MAT_ID` copy branch in the frozen one-token/ten-layer execution. This is consistent with the branch being conditional on host-resident weights being copied into a backend split; it cannot be assumed to observe a normal offloaded layer.

Per the approved stop conditions, the experiment did not search other layers, request tensors, add synchronization, instrument CUDA, alter graph segmentation, or widen into allocator/backend work. The 185-line one-file prototype remains dirty in the isolated llama.cpp worktree for audit and has not been committed. The pristine pinned checkout remains untouched.

## Decision

Observer v2 has not restored parity-safe routing evidence because it produced no IDs. Gate 4 remains closed, `live_cache_enabled=false`, prior traces remain quarantined, the historical `93.28%` claim remains withdrawn, and corpus collection remains stopped. A new implementation direction requires explicit approval; no automatic expansion is recommended from this failed boundary.

The machine-readable stop record is `C:\models\expertflow\runs\trace-observer-v2-prototype\stop-summary.json`.
