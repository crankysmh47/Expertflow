# ExpertFlow Live TPS Demo Design

## Goal

Provide one judge-readable Windows command that measures stock and ExpertFlow Q6 decode throughput live using the frozen matched protocol, while keeping the authoritative ten-pair result and all quality limitations unchanged.

## Modes

- `Demo`: one matched 512-token pair for recording. It prints each mode before launch and its result immediately after completion.
- `Judge`: three alternating matched 512-token pairs, followed by means, dispersion, VRAM, wall time, and an evidence path.
- The existing ten-pair command remains the authoritative reproduction protocol.

## Contract

- Use the exact Q6 model, patched runtime, prompt, context, `-ngl`, CPU-MoE, threads, seed, sampling, CUDA-graph, and static-precompute settings already frozen in `deployments/max-performance.json`.
- Stock starts with all ExpertFlow placement variables cleared. ExpertFlow uses layers `0,1,2,3,4,5,6,7,8,9,15,20` and precomputes them.
- Run each mode in a fresh process and alternate order when multiple pairs are requested.
- Display decode TPS separately from prompt TPS, process wall time, and process-owned peak VRAM.
- Store raw logs, measurements, summary JSON, and CSV outside the repository by default.
- A single pair reports sample standard deviation as unavailable, not zero.
- Record response hashes without claiming cross-mode token parity. Preserve the strict PPL confidence-gate failure and single-system scope.
- Fail visibly when the model or runtime is absent. Print the offline replay command as the supported fallback; never relabel replay as a live benchmark.

## Recording Flow

1. Run a preflight that names the model, runtime, GPU, tokens, pair count, and exact placement.
2. Show `STOCK` loading and its live TPS result.
3. Show `EXPERTFLOW` loading, its emitted placement, and its live TPS result.
4. Hold on the side-by-side table and saved evidence path.
5. Continue into the animated slideshow for the problem, compiler insight, Codex process, result, limitations, and close.

## Verification

- Source-contract tests freeze the matched arguments and explicit progress/result labels.
- Unit tests cover one-pair summary behavior and incomplete pairs.
- Run the one-pair 512-token demo on the verified RTX 5060 Ti system, confirm cleanup and idle VRAM, then retain the result as rehearsal evidence only.
- Run applicable tests, rebuild the deterministic release, and keep the branch clean before handoff.
