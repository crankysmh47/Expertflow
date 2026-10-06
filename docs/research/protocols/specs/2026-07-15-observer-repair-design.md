# Non-Perturbing Router Observer Design

Status: approved by the attached 2026-07-15 tracing-repair directive

## Goal

Make router observation exactly transparent before any further real-model trace collection or live-cache work. The first engineering block isolates the earliest perturbing behavior. It does not optimize the current callback after callback registration itself is shown to fail.

## Preserved boundary

- Preserve audit commit `c41b9394c0443a66b3b486936d128c034ea3a4d7`.
- Keep protected Observatory commit `d846bdfcb1980dfc44d9f951e2824f58429f16d7` clean.
- Keep pinned llama.cpp checkout `a7312ae94f801fc9c6786dc56e38df57b964f697` clean.
- Keep Gate 4 closed and `live_cache_enabled=false`.
- Do not collect more corpus traces, implement cache code, add MTP, or weaken exact parity.

## Evidence quarantine

Create one checked-in evidence-status manifest. Every real-model trace produced through the current evaluation callback is labeled `trace_v1_perturbing` and `excluded_from_final_claims=true`, including the original Q4 probe, stratified CUDA/Vulkan, 40-conversation physical-feasibility, held-out, deadline, and Gate 3 diagnostic trace roots. Derived locality, static, LRU, session, oracle, and deadline results inherit the quarantine.

Quarantine preserves files and provenance; it never deletes or rewrites artifacts. These artifacts may be cited only as historical/diagnostic evidence that led to the observer repair. The checked-in eight-event replay fixture and synthetic tests remain valid for deterministic parser/simulator reproduction, but they cannot support real-model locality or cache-policy claims.

The public documentation must explicitly withdraw the multi-domain `93.28%` result from final recommendation use until it is reproduced from a parity-safe replacement corpus.

## Staged minimization matrix

Use the same frozen general, code, and translation prompts, greedy decoding, `-n 16`, `-ngl 10`, 12 threads, one-token batch/ubatch, model hash, GPU, and runtime. Each tested variant receives three baseline runs and three observer runs per domain. Compare complete prompt/generated token arrays, ordered router selections where the variant exposes them, event completeness, causal ordering, wall duration, sampled GPU/host memory, and persistent processes.

The variants are staged rather than implemented all at once:

| Variant | Behavior | Continue condition |
| --- | --- | --- |
| T0 | Tracing code compiled; callback not registered | Establish canonical output |
| T1 | Callback registered; every invocation performs no mutation/readback and returns `false` on `ask=true` | Only implement T2 if exact token parity passes |
| T2 | Increment one fixed counter only | Only implement T3 if parity passes |
| T3 | Record token/layer metadata into fixed-capacity storage | Only implement T4 if parity passes |
| T4 | Capture selected expert IDs into fixed-capacity storage | Only implement T5 if parity passes |
| T5 | Capture expert IDs and weights | Only implement T6 if parity passes |
| T6 | Defer serialization until after `llama_decode()` | Compare with T7 |
| T7 | Current callback and synchronous serialization | Historical perturbing control |

If T1 fails, callback registration/scheduler interception is the root boundary. Stop T2-T7 implementation in this callback and recommend a different instrumentation boundary. This prevents spending the timebox on allocations or ring buffers that cannot repair scheduler-level perturbation.

## T0/T1 diagnostic interface

Extend only the separate `native/router_probe` executable with diagnostic `--trace-mode` values:

- `full`: current callback/readback/serialization behavior; remains the default when `--trace` is used so existing commands do not silently change meaning.
- `empty`: register a callback whose body has no allocation, I/O, formatting, tensor access, synchronization, or state mutation and returns `false` for every scheduler `ask`.
- `disabled`: do not register a callback; equivalent to `--no-trace`.

The empty mode writes no router events and must not be treated as a valid trace. Its only output is the token artifact and process measurements. Invalid combinations fail before model load. The production/default release remains tracing-disabled.

## Root-cause criterion

The pinned scheduler has two different execution paths:

- no callback: compute each backend split asynchronously as one graph;
- callback registered: execute graph views, synchronize the backend after every view, and call the observer.

T1 isolates this branch without host readback or callback work. If T1 changes tokens, the causal boundary is callback registration and scheduler interception, not JSON formatting, heap allocation, file I/O, locks, or `ggml_backend_tensor_get` in the callback body.

## Repair direction after T1 failure

Do not weaken parity and do not call the synchronized evaluation callback “non-perturbing.” Investigate a boundary that does not select the callback scheduler path. Candidate order:

1. A post-graph/decode hook that observes stable router results after normal graph completion without changing graph partitioning.
2. A backend-native deferred device-to-host copy recorded on the producer stream and consumed only after normal completion.
3. A narrowly isolated llama.cpp diagnostic build or backend event hook, never a mutation of the pinned clean checkout.

Any candidate must first prove token parity on general, code, math/reasoning, and translation, then pass a frozen 10–20 prompt suite with three repetitions per mode before collecting a pilot corpus.

## Safety and failure handling

Future fixed-capacity storage must be allocated before decode, use explicit append bounds, canaries, and overflow failure, and never silently overwrite/drop events. Callback code may not allocate, grow containers, format strings, write files, lock, mutate tensors/graphs, or read device data synchronously. Serialization occurs after the model step.

## Cutoff and deliverable

This block ends when the first failing variant is reproduced and documented, or when T1 passes and the next variant is justified. If no prompt-transparent boundary can be established promptly, live caching remains closed, the multi-domain 93.28% claim stays withdrawn, and tracing sensitivity is reported transparently.

The deliverable is a committed quarantine manifest, T0/T1 test evidence, append-only ledger, and either a verified path to T2 or a root-cause report recommending a different boundary.
