# T2 Projected-State Policy Design

## Objective

Run one final bounded layer-24 predictive-policy experiment. Preserve the
existing exact 34-slot T2 runtime and change only candidate admission:
predictions are filtered against the projected reactive cache state after the
current token's authoritative layer-24 demands have completed normal reactive
admission, eviction, and LRU updates.

After this experiment the ExpertFlow runtime is frozen regardless of outcome.

## Diagnosis being tested

The unconditional T2 sidecar produced ready-useful transfers, but every
sidecar-served demand was already a reactive hit in the paired baseline. The
candidate was absent when prediction ran, then entered the reactive cache
during the source token.

The final policy tests whether filtering against the post-source-token
projected state removes those redundant transfers and redirects the one-copy
budget toward experts that remain absent.

## Architecture

The canonical layer-24 router callback already has:

- the eight true current-token expert IDs;
- the frozen temporal candidate ranking for token `t+1`;
- the current protected 32-slot reactive state;
- the cache planner and commit logic used by live execution.

For each decode token `t`:

1. Copy the current layer-24 reactive state.
2. Run the unchanged reactive planner on the eight authoritative experts for
   token `t`, using the current force-evict configuration.
3. Commit the plan to the copied state only.
4. Traverse the frozen width-16 temporal candidates in order.
5. Select the first supported candidate absent from the projected copied
   state.
6. Enqueue at most one transfer through the existing T2 sidecar path.

The real reactive state, LRU order, mappings, loads, misses, and event
accounting remain authoritative and unchanged. The projection is advisory and
must never be committed to live state.

## Scope and invariants

- layer 24 only;
- reactive slots 0-31 unchanged;
- speculative slots 32-33 unchanged;
- one contiguous 34-slot packed tensor;
- one asynchronous transfer per decode token;
- frozen temporal predictor, weights, support, width, and ordering;
- no retraining or retuning;
- no promotion into reactive slots;
- unchanged exact blocking fallback;
- unchanged `MUL_MAT_ID`;
- no CUDA kernel, operation, graph, allocator, placement, quantization, or
  repacking change;
- disabled by default behind the existing T2 feature flag.

The event log adds the policy identity and projected candidate decision. It
must distinguish:

- candidate already present before projection;
- candidate admitted by the projected current-token plan;
- candidate still absent and transferred;
- no projected-absent candidate;
- unsafe sidecar slot.

## Correctness

The pure policy must fail closed when:

- authoritative IDs are invalid;
- the current state cannot produce a valid exact reactive plan;
- the projected commit cannot be validated;
- candidate IDs are invalid;
- more than one transfer would be selected;
- projection mutates the live state.

Live validation requires exact reactive-versus-predictive:

- prompt tokens;
- generated tokens;
- router-selected expert IDs and ordering;
- routing weights where present;
- event counts and causal order;
- deterministic repeated runs;
- normal reactive mappings and state transitions for demands not served by the
  sidecar.

## Measurements

Use the same general, code, and translation prompts, `-ngl 10`, Q4 model,
threads, token count, warmup, and three measured repetitions as T2.

Report:

- actual baseline blocking misses covered by sidecar use;
- ready-useful, late-useful, wasted, no-candidate, and unconsumed-final
  prefetches;
- reactive and sidecar blocking time;
- prompt TPS, decode TPS, end-to-end time, TTFT, and p50/p95 latency;
- staged, queued, CUDA-event, and host queue-to-ready timing as separate
  classes;
- bytes moved and wasted bytes;
- exact projected and measured arena bytes;
- peak VRAM, host memory, settled cleanup, and process exit;
- repetition values, means, variance, and per-domain results.

## Outcome rule

The experiment succeeds as an engineering primitive when exactness, bounded
memory, safe cleanup, and protected reactive behavior pass.

It succeeds as a performance policy only when it prevents real paired-baseline
blocking misses and the resulting blocking reduction is not erased by
prediction, transfer, remapping, or instrumentation overhead.

Regardless of outcome:

- commit the implementation and evidence separately;
- mark runtime development frozen;
- do not start another predictor, multi-layer prefetch, MTP, RL, cache-size
  sweep, confidence sweep, or concurrency sweep;
- proceed to release integration, final benchmark curation, documentation, and
  submission assets.
