# Router Observer Perturbation Isolation

Status: **FIRST FAILING BOUNDARY IDENTIFIED; OBSERVER NOT REPAIRED**

Recorded: 2026-07-15 PKT

Pinned llama.cpp: `a7312ae94f801fc9c6786dc56e38df57b964f697`

Diagnostic probe commit: `e5fc4ff`

Default/live state: `live_cache_enabled=false`

## Outcome

The smallest failing behavior is not callback registration, callback volume, fixed-buffer mutation, layer-name parsing, or host tensor readback. It is requesting the router tensor by returning `true` from the evaluation callback. That request makes the pinned scheduler end a graph view at every `ffn_moe_topk-*` tensor and synchronize the backend before calling the observer.

A boundary-only callback that reads zero tensor bytes reproduces the same deterministic token changes as selected-ID capture and the historical full tracer:

- Code: generated token 0 changes from `108` to `5676`.
- Translation: generated token 2 changes from `45518` to `676`.
- General chat remains identical.

The current callback boundary cannot be repaired with a ring buffer, deferred JSON formatting, or removal of `ggml_backend_tensor_get`, because the perturbation occurs before host readback. Tracing must move to a boundary that preserves the normal scheduler path.

## Evidence quarantine

All current callback-derived real-model traces are labeled `trace_v1_perturbing` in `configs/trace-evidence-status.json`. They are preserved for audit and excluded from final locality, static, LRU, session, oracle, deadline, recommendation, and Gate 4 claims. The historical `93.28%` result is withdrawn pending parity-safe recollection. No replacement corpus was collected during this stage.

The checked-in replay fixture remains valid only for offline parser/simulator reproduction. It does not support a real-model policy claim.

## Controlled variants

Every variant used the exact clean CUDA 12.8 runtime, Q4 model, greedy sampling, 16 generated tokens, ten GPU layers, 12 threads, one-token batch/ubatch, and the same general, code, and translation prompts. Each domain received three callback-disabled and three candidate runs. All candidate and baseline repetitions were internally deterministic, and no model process persisted.

| Variant | Hot-path behavior | General | Code | Translation | Decision |
| --- | --- | --- | --- | --- | --- |
| T0 | Callback not registered | Baseline | Baseline | Baseline | Canonical output |
| T1 | Empty callback; always returns false | PASS | PASS | PASS | Registration alone is safe |
| T2 | Increment one preallocated counter; return false | PASS | PASS | PASS | Callback volume/state increment is safe |
| T3 | Write token index/layer ID to fixed storage; return false | PASS | PASS | PASS | Metadata capture is safe |
| T3.5 | Return true at selected-ID tensor; count observation; no readback | PASS | **FAIL at output 0** | **FAIL at output 2** | First failing behavior |
| T4 | Same boundary plus eight-I32 preallocated host copy | PASS | **FAIL at output 0** | **FAIL at output 2** | Readback is not required for failure |
| T5–T7 | Weights, flush variants, historical full logging | Not run | Not run | Not run | Stopped after first failing boundary; historical T7 already fails |

T1 median differences relative to its paired disabled runs range from `-0.03%` to `+0.42%`. T2 ranges from `-0.87%` to `+0.38%`; T3 from `-1.14%` to `-0.04%`; boundary-only from `-0.36%` to `+0.09%`; T4 from `-0.35%` to `+0.21%`. These short-run wall-time differences are descriptive and not performance claims. The important result is exact token parity.

## Buffer and stability checks

- T2 callback ask counts are stable: general `140,132`, code `150,708`, translation `142,776`.
- T3 records exactly `1,590`, `1,710`, and `1,620` token/layer events respectively, with zero overflow and intact front/back sentinels in all nine candidate runs.
- T3.5 observes the same `1,590`, `1,710`, and `1,620` selected-ID boundaries without reading tensor data.
- T4 captures the same event counts; every expert ID is in `[0, 127]`, all event/value sentinels remain intact, and no overflow or tensor-contract error occurs.
- Every T4 token artifact exactly matches the corresponding historical full trace-on artifact. This independently reduces the failure to the common requested-tensor scheduler boundary.
- Historical full-trace repeats provide ordered router-selection repeatability. The minimized T1–T3 modes cannot compare expert selections by construction because they intentionally do not observe expert IDs; T4 validates captured IDs in-process but does not serialize them, to avoid adding another hot-path variable.

## Source causality

The diagnostic callbacks are isolated in `native/router_probe/main.cpp`:

- empty: line 418;
- counter: line 422;
- metadata: line 429;
- boundary-only: line 450;
- selected IDs: line 461.

In pinned `ggml/src/ggml-backend.cpp`, line 1677 uses whole-split asynchronous compute when no callback exists. With a callback, lines 1688–1695 ask which tensor is needed, line 1706 synchronizes the backend after the resulting graph view, and line 1708 performs the observation callback. T1–T3 always return false, so each backend split remains one view. T3.5/T4 return true at every router tensor, creating the perturbing view boundaries.

This is a source-supported causal inference combined with measured token evidence. It does not claim that synchronization is generally invalid; it establishes that this model/runtime's exact output changes when these graph cuts are introduced.

## Decision

- Stop T5–T7 and all additional corpus collection.
- Keep Gate 4 closed and `live_cache_enabled=false`.
- Do not narrow the final claim to the general prompt merely because it passes; the observer must satisfy the broader fixed suite.
- Do not attempt to repair this callback with allocation/I/O/ring-buffer changes.
- Move the next bounded investigation to a post-graph/decode hook or backend-native deferred observation that does not set `callback_eval` and does not partition the graph.
- Require exact token and router parity on general, code, math/reasoning, translation, then 10–20 frozen prompts with three repetitions before collecting even a pilot replacement corpus.

## External artifacts

Root: `C:\models\expertflow\runs\trace-observer-repair`

- `isolation-summary.json`: consolidated machine-readable decision.
- `matrix-summary.json`: T0/T1 empty callback.
- `matrix-summary-counter.json`: T0/T2 counter.
- `matrix-summary-metadata.json`: T0/T3 metadata.
- `matrix-summary-boundary.json`: T0/T3.5 boundary-only.
- `matrix-summary-ids.json`: T0/T4 selected IDs.
- `matrix/*`, `matrix-counter/*`, `matrix-metadata/*`, `matrix-boundary/*`, `matrix-ids/*`: commands, logs, tokens, timings, and memory per run.
- `commands.jsonl`: append-only decisions, failures, corrections, and measurements.

The final diagnostic probe is 2,958,259 bytes, SHA-256 `0b9ea3cc26663414eb3ba523e49f2ef19da9c85d1420b9047f85271c8340a311`.

Checkpoint verification passed 88 tests, the 8-event/64-demand replay, a fresh probe build/CLI contract, seven JSON files and the append-only JSONL ledger, all recorded artifact/source hashes, `git diff --check`, protected/source identity and cleanliness, and no-cache environment/process checks. `isolation-summary.json` is 11,577 bytes with SHA-256 `02ffe8fe1eca25fd36d4b73a82ca4364fb0f441d628dee2b1b488cd250f2becc`; `verification.json` is 1,934 bytes with SHA-256 `a96cc5363700af2bcf59aa61aa6341eddd272ef09a0432f449c6697a9fc30bee`. The 38-record pre-commit ledger is 10,608 bytes with SHA-256 `083da05301678feccdaa3bdd5b363a14cffb328251eb27d69be01347a99d9ff8`.
