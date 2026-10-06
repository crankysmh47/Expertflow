# Canonical Observer Runtime Smoke Plan

**Goal:** Decide whether the restored graph-segmenting Observer v1 is a competent, deterministic canonical ExpertFlow runtime using the fixed seven-task smoke suite.

## Guardrails

- Preserve Observatory `d846bdf`, Observer v2 `8c3cef0`, and pinned llama.cpp `a7312ae`.
- Keep cache disabled; do not implement caching or redesign the observer.
- Compare normal and full-observer modes from one probe binary and source baseline.
- Treat wall time as diagnostic metadata, not a performance claim.
- Quarantine all prior `trace_v1_perturbing` data.

## Execution

1. Preserve the failed Observer v2 prototype in a named stash and record its recovery command.
2. Build the restored Observer v1 probe in an isolated runtime. Add only post-generation text serialization needed for objective smoke validation.
3. Record source/model/binary hashes, compiler/CUDA/driver details, and deterministic runtime parameters.
4. Confirm Observer Mode O token and routing determinism on one repeated prompt.
5. Run exactly seven fixed short tasks in Mode N and Mode O, capturing tokens, Mode O routing, memory, duration, and process cleanup.
6. Validate code locally, exact arithmetic, strict JSON, translation facts, and bullet constraints; repeat only an ambiguous task once.
7. Produce concise JSON and Markdown evidence and decide pass/fail against the supplied acceptance criteria.
8. On pass, pin the canonical runtime and begin only the small `trace_v2_canonical_segmented` pilot.

