# Bounded Live-Cache Spike Design

**Status:** Approved by the user for bounded execution on 2026-07-15. The protected Observatory remains the submission floor.

## Objective

Prove, without weakening the unchanged runtime path, that one true-router-selected Gemma 4 Q4 expert object can move from host storage into a reusable CUDA slot, execute from that preallocated slot, be replaced repeatedly, and preserve exact prompt tokens, generated tokens, and router selections.

This is a correctness and physical-execution experiment. It is not a runtime-speedup claim, a CUDA deadline claim, a production cache, or permission to implement prediction, asynchronous prefetch, MTP, learned policy, a broad allocator rewrite, or the projected 96-slot-per-layer allocation.

## Protected floor and isolation

- Protected commit: `d846bdfcb1980dfc44d9f951e2824f58429f16d7`.
- Annotated tag: `observatory-floor-2026-07-15`.
- Protected branch: `codex/expertflow-stage0`; it receives no live-cache changes.
- Live branch: `codex/live-cache-blocking-spike` in the C-drive worktree `C:\models\expertflow\worktrees\live-cache-blocking`.
- All generated native sources, builds, logs, and measurements stay under `C:\models\expertflow`; the D drive is not used.
- The protected CPU-only judge replay and the self-contained Observatory report must remain reproducible before and after every native-runtime gate.

## Evidence correction and verdict

The practical non-oracle evidence is valid for a bounded spike because the static-96 residents were derived exclusively from 31 parity-safe `train` conversations and scored on eight untouched conversations: four `validation` and four `test`. The evaluator counts residents from `training_events` and scores `evaluation_events` separately. The decision therefore changes from `NO-GO` to `CONDITIONAL-GO-FOR-BOUNDED-SPIKE`.

This correction does not turn the 9.0257% cold-byte reduction over conversation-reset LRU into a stronger result. It only establishes that the reported static-96 comparison is genuinely held out. The default remains `live_cache_enabled=false`.

## Gate sequence

### Gate 1: protected reproduction and provenance

Capture the exact Git refs, hashes of every runtime/build binary used, model hash, NVIDIA driver/GPU information, compiler/tool versions, relevant non-secret environment variables, commands, timestamps, durations, outputs, and artifact hashes. Re-run the clean judge fixture and regenerate the physical Observatory from the protected ref. No installer runs before this evidence is complete.

### Gate 2: supported CUDA toolchain

Install Visual Studio 2022 Build Tools side by side with Visual Studio 2026. Select Desktop development with C++, the v143 MSVC toolset, and a compatible Windows SDK. Install CUDA Toolkit 12.8 with `nvcc`; do not replace the working NVIDIA driver. Reuse CMake and Ninja.

The gate requires a VS 2022 developer environment in which `cl` resolves to v143, `nvcc --version` reports CUDA 12.8, CMake detects MSVC and CUDA, NVIDIA `deviceQuery` builds and passes, and the existing standalone pinned-transfer benchmark remains within a documented variance band of its prior 3,346,048-byte pinned result (0.234016 ms p50 / 0.234272 ms p95). Any variance decision must include raw samples and system state rather than silently changing the baseline.

### Gate 3: exact unmodified llama.cpp baseline

Use a clean Git checkout at `a7312ae94f801fc9c6786dc56e38df57b964f697`. Configure and build CUDA without ExpertFlow source changes. Record the source commit/tree, CMake cache, complete configure/build commands, compiler identities, binary hashes, and test output.

Run model load, CPU inference, ten-layer CUDA offload, tracing off/on where the pinned source exposes the existing trace boundary, prompt/generated token parity, router-event parity, peak VRAM, basic throughput, llama.cpp tests, and all 87 ExpertFlow tests. If the unmodified source cannot expose the existing router trace without modification, validate tokens and runtime first, record that instrumentation limitation explicitly, and do not claim router-event parity until the trace-only patch is separately proven neutral. No cache implementation begins unless known-good behavior is reproduced.

### Gate 4: one-layer blocking proof

The integration boundary is the existing `GGML_OP_MUL_MAT_ID` selected-expert copy path in `ggml/src/ggml-backend.cpp`. The feature is disabled unless an explicit ExpertFlow environment flag identifies one layer, slot layout, and JSONL movement log. Unset or invalid configuration executes the original code path byte-for-byte.

The model selects eight experts per token. The exact unchanged `MUL_MAT_ID` operation therefore cannot execute from only one or two total physical expert slots. The bounded working interpretation is eight physical slots for one selected layer: seven fixed slots plus one replaceable proof slot, with a controlled token sequence whose demand changes by at most that one slot. If no measured sequence satisfies that condition, or if the user requires one or two total slots, stop before implementation because satisfying it would require the prohibited graph/kernel restructuring.

The measured sequence exists entirely in the training split. In `train-general-08`, layer 1, token indices 62 through 111, experts `0,13,84,88,92,95,124` remain fixed while the eighth expert rotates among `28,40,66`. This gives 50 decode events for repeated one-slot replacement without choosing a proof sequence from validation or test data.

Each physical slot is a coordinated set of the existing gate, up, and down Q4 tensor slices. Loading an object performs three direct host-to-device range copies into already allocated component destinations sharing one slot index. It does not dequantize, convert, concatenate, or repack. The measured object contract is 3,345,412 encoded bytes and 3,346,048 aligned bytes.

At the selected layer, the scheduler reads the authoritative logical IDs after the router completes, computes the controlled fixed-plus-replaceable mapping, blocks until all missing component ranges are resident, supplies physical slot IDs only to the expert matrix operations, and preserves/restores the logical IDs for routing weights, callbacks, and parity evidence. Any scale, bias, adapter, or later consumer that needs logical IDs and cannot remain isolated makes the feature fail closed for this model.

The movement JSONL record includes schema version, run/request/token index, layer ID, logical expert ID, packed/aligned/copy bytes, source tensor/address/state, destination slot/address, queue/start/end CUDA timestamps where the backend supports them, host timestamps, blocking duration, reuse, eviction and reason, allocation totals, and synchronization events. A load is reported as one object transaction with its three component copies.

### Gate 5: proof acceptance

Pass requires all of the following:

- Exact prompt tokens, generated tokens, and router selections versus the unchanged CUDA baseline.
- Direct execution from preallocated Q4 slot tensors with no per-load repacking.
- Forced misses and repeated swaps, including slot reuse and deterministic replacement evidence.
- Stable allocations across repeated fresh and same-process runs, with no stale slot, lifetime, race, corruption, synchronization, or growth symptom.
- Live blocking transfer latency reasonably consistent with the standalone CUDA microbenchmark after clearly accounting for runtime contention and synchronization.
- Feature-off behavior restores the original path and binary-level configuration defaults remain off.
- llama.cpp baseline tests, all 87 ExpertFlow tests, clean replay, and Observatory regeneration remain green.

The proof is committed separately before any expansion.

## Expansion boundary

Only a passing proof permits a later design for a few layers and a minimal per-layer slot table. Capacity grows progressively and is measured for actual allocation, fragmentation, and VRAM headroom. Static residents must be training-only and evaluated on untouched held-out prompts against a same-runtime reactive blocking baseline.

Prediction and asynchronous CUDA overlap are a later gate after an exact static live cache. MTP remains excluded.

## Stop conditions

Stop runtime work and return to submission/demo work if the clean pinned build fails parity; movement requires per-load Q4 repacking; exact execution requires a broad graph, kernel, or allocator rewrite; parity fails; corruption, unstable synchronization, or allocation growth appears; the feature cannot remain isolated and disabled by default; or the work threatens the July 18 technical cutoff or protected floor.

## Time box

Use a four-hour total bound from the protected-floor resume: at most 90 minutes for provenance, toolchain, and the unmodified build gate, then at most 150 minutes for the exact one-layer proof and evidence. Stop earlier on any gate failure. Documentation, evidence hashes, tests, and commits are part of the time box rather than deferred cleanup.
