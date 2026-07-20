# ExpertFlow final Q6 runtime design

Status: accepted by the user on 2026-07-18.

## Product boundary

ExpertFlow will preserve Gemma's authoritative router/top-k and surrounding layer computation on their stock backend. Only the routed-expert operation island may move to CUDA. CPU memory remains authoritative for complete expert bundles; CUDA holds a bounded, coherent set of packed bundles. True router selections remain authoritative and exact blocking fallback precedes any reuse of the existing predictor.

## Placement approaches

1. **Selected:** use existing scheduler-supported pre-scheduling node/tensor backend assignment so final IDs and activation cross to a CUDA expert island and the expert result crosses back through ordinary scheduler copies.
2. **Only fallback:** explicitly construct the same expert subgraph boundary using existing scheduler copy/allocation APIs.
3. **Rejected:** whole-layer placement. Both historical variants changed numerical ordering and are closed.

No new MoE kernel, scheduler replacement, post-allocation mutation, manual split repair, or whole-layer relocation is permitted. Failure of both bounded narrow approaches produces `NARROW PLACEMENT STOP`.

## Gate sequence

The runtime advances only through: truthful Q6 baseline; Q4 static narrow island; Q4 reactive cache; quantization-independent bundle metadata; Q6 static island; whole-model Q6 reactive product; existing predictor comparison; developer CLI packaging. Exact tokens and ordered router IDs are hard gates. Later stages are not started after a terminal stop.

## Evidence and rollback

ExpertFlow work is isolated on `codex/q6-runtime-final` from `33bc3f5`. Any llama.cpp experiment uses a separate source worktree and build directory. Successful milestones are committed separately. Failed placement implementation code is discarded; only concise failure evidence and a reproducible source-contract/test artifact may be committed. Existing Q6 characterization and the earlier `PLACEMENT STOP` are immutable inputs.
