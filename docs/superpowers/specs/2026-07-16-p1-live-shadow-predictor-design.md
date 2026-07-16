# P1 Live Shadow Predictor Design

## Objective

Integrate the frozen expanded-corpus B2 predictor into the canonical runtime in
shadow mode for the single MoE transition from layer 23 to layer 24. P1
produces candidates, measures prediction latency and quality, and changes no
cache, transfer, graph, routing, sampling, or tokenization behavior.

The frozen predictor is commit `6bc8eb68`, source-normalized,
phase-separated B2 transition tables, width 12, and observed-support
admission. The sealed test is not rerun or used for further selection. Its
selection lock existed before the sole test evaluation but was committed
afterward; this procedural exception remains disclosed.

## Branch and default-state contract

- ExpertFlow branch: `codex/predictive-shadow-integration`, based on
  `7184fb7`.
- llama.cpp branch: `codex/predictive-shadow-integration`, based on
  `b910bc37`.
- `live_cache_enabled=false` remains the default.
- Predictor shadow mode is disabled by default and requires an explicit
  artifact path and output path.
- P1 moves no expert weights and performs no predictor-driven cache action.

## Versioned binary artifact

ExpertFlow exports the selected pickle into a runtime-neutral little-endian
binary artifact. The runtime never imports Python or pickle.

The fixed header contains:

- eight-byte magic and unsigned format version;
- fixed-width model and runtime identifiers;
- source layer 23, target layer 24, expert count 128, source width 8, and
  candidate width 12;
- scoring identifier `source_normalized`;
- table identifier `phase_separated`;
- admission identifier `observed_support`;
- tie-break identifier `ascending_expert_id`;
- training manifest SHA-256 and predictor-configuration SHA-256;
- payload byte count and SHA-256.

The payload stores prefill and decode data for the 23-to-24 transition:

- for each of 128 source experts, 128 float64 normalized target scores;
- an explicit 128-bit observed-support mask for each source expert;
- a 128-element fallback frequency score vector for layer 24.

Float64 preserves the offline score arithmetic closely enough to require exact
candidate ordering. The exporter emits deterministic fixtures containing phase,
eight source IDs, expected top-12 IDs, and their scores. Runtime tests compare
against these fixtures.

The loader reads once during initialization into fixed-size arrays. It rejects
bad magic/version, identifiers, dimensions, layers, candidate width, checksum,
truncation, trailing bytes, non-finite or negative scores, and malformed
support masks. There is no allocation per prediction.

## Runtime scoring

For each source event at layer 23:

1. Read the eight already-materialized true source IDs.
2. Require a valid phase for the current evaluation call.
3. Select the prefill or decode table.
4. Sum each source expert's normalized float64 target vector.
5. If every score is zero, use the layer-24 fallback vector.
6. Admit only candidates with positive observed transition support.
7. Rank by descending score, then ascending expert ID.
8. Return exactly 12 candidates or fail if fewer than 12 supported candidates
   exist.

The scorer is a private fixed-capacity component independent of the cache
planner. It cannot mutate residency, mappings, generations, or transfer state.

## Explicit phase lifetime

The probe is the highest boundary that reliably knows whether a one-token
`llama_decode` call is prefill or decode. It sets phase immediately before the
call and clears it immediately afterward, including failure paths.

A narrow llama API accepts only `prefill`, `decode`, or `unset`. Each set
operation advances a generation. The scheduler snapshots the active generation
for the evaluation and rejects:

- unset or invalid phase;
- a generation reused across calls;
- contradictory set operations during one call;
- a phase that remains active after the caller clears it.

The phase is read-only to the predictor and is not derived from token counts,
positions, batches, or scheduler heuristics.

## Shadow capture and reconciliation

Layer 23 prediction occurs after its true IDs are available at the existing
host-ID materialization boundary. The twelve predicted IDs, scores, source IDs,
phase, phase generation, and prediction duration are stored in one pending
fixed-capacity record.

When layer 24's true IDs become available in the same evaluation call, the
runtime reconciles the pending prediction and appends a completed record.
Missing, duplicate, out-of-order, cross-generation, or incomplete transitions
fail closed.

Each deferred JSONL record contains:

- run identifier, forward/evaluation identifier, phase, and phase generation;
- source and target layers;
- eight source IDs;
- twelve predicted IDs and scores;
- eight actual target IDs;
- recall@8 and recall@12 hit counts and normalized contributions;
- prediction latency in nanoseconds;
- artifact format version, payload SHA-256, and configuration SHA-256.

The capture path performs no file I/O, string formatting, heap allocation,
additional tensor request, callback, or synchronization beyond the existing
reactive ID materialization. Fixed-capacity overflow is fatal. Serialization
occurs at scheduler destruction.

## Validation ladder

### S0 - compiled and disabled

Require exact parity with `b910bc37` and no predictor initialization.

### S1 - enabled, artifact validation only

Load and validate the artifact, set and clear phase, but record no predictions.
Require exact parity and stable memory.

### S2 - deterministic scorer fixtures

Run native tests for prefill/decode selection, known source sets, candidate
ordering and scores, invalid artifacts, missing/invalid/stale phase, reset
behavior, and repeated deterministic calls.

### S3 - live layer 23 to 24 shadow

Run general, code, and translation prompts, then the seven-task smoke suite.
Require:

- exact shadow-off/on prompt and generated tokens;
- exact ordered router selections;
- live candidates equal offline candidates for every identical input;
- correct phase and transition counts;
- deterministic repeated records excluding measured duration;
- no graph segmentation change;
- no cache mutation or predictor-driven transfer;
- stable CPU/GPU memory and complete process cleanup.

Report live p50/p95 prediction latency, recall@8/12 overall and by phase,
candidate-support failures, and runtime overhead against the same binary with
shadow disabled.

## Stop conditions

Stop before P2 if any of the following occurs:

- offline/live candidate or score ordering differs;
- exact runtime parity fails;
- phase cannot be scoped to one evaluation call;
- another tensor request, callback, synchronization, or graph split is needed;
- the predictor changes cache or transfer state;
- memory grows persistently;
- fixed-capacity safety cannot be guaranteed;
- the implementation expands beyond the single 23-to-24 transition.

If S3 passes, commit the runtime integration and ExpertFlow evidence separately,
pin both binary and artifact hashes, publish a concise parity/latency report,
and issue but do not implement the P2 asynchronous-transfer design.
