# ExpertFlow Product Release Design

## Boundary

Productize the frozen Gemma 4 26B A4B Q6 result as a small, evidence-driven Windows CLI and offline release. The runtime architecture stays frozen: twelve full 128-expert CUDA banks at layers `0,1,2,3,4,5,6,7,8,9,15,20`; no reactive cache, eviction, prediction, scheduler redesign, or new performance claim.

## Architecture

The Python package owns deployment manifests, evidence verification, environment inspection, command construction, evidence replay, and report rendering. It never embeds the GGUF and never assumes a private path: model and binary locations resolve from command arguments, environment variables, then optional manifest hints. A deployment JSON is the sole interface between `optimize`, `run`, `serve`, `compare`, and agentic examples.

Recorded results remain immutable inputs. Live commands label hardware inspection and new measurements separately from replayed evidence. Unsupported throughput or context profiles remain explicit unavailable profiles rather than projections presented as measurements.

## Components and data flow

1. A frozen product-result manifest records commits, hashes, build flags, winning command, quality status, and predictive-cache stop evidence.
2. `expertflow doctor/profile/optimize/run/serve/compare/demo` load and validate that manifest through focused release modules.
3. Server benchmark and context-frontier harnesses emit machine-readable raw results. Only successful repeated measurements may become throughput or context deployments.
4. The existing Observatory is extended into an offline product dashboard driven only by committed JSON.
5. A duplicate-safe release assembler copies an allowlist, scans it for private paths and credentials, writes SHA-256 inventory, and creates the ZIP only after verification.

## Error handling

Commands return stable JSON with `status` equal to `pass`, `warning`, or `failure`. Model/hash or runtime identity mismatches fail before launch. Missing CUDA/model hardware does not break `demo --replay`. A requested profile without measured evidence fails clearly rather than falling back silently.

## Verification

Development follows test-first CLI and manifest contracts. Final checks cover applicable tests, model-free replay, evidence hashes, offline dashboard loading, release inventory, path and credential scans, git cleanliness, and process cleanup. Historical temporal source-contract tests are excluded because they target a different preserved llama.cpp branch; that exclusion is recorded rather than hidden.

