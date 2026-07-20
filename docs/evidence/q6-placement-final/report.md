# ExpertFlow final bounded Q6 placement sprint

## Outcome

The placement experiment produced a large, reproducible throughput result but failed the mandatory held-out quality confidence-bound gate. The terminal verdict is **QUALITY STOP**. No dynamic cache, predictor, scheduler redesign, product command, or deployment default was added.

## Winning measured placement

The bounded greedy search retained full 128-expert static CUDA shadows for layers `[0,1,2,3,4,5,6,7,8,9,15,20]`. The shadows use identity logical-to-physical mapping, keep CPU expert tensors authoritative, and allocate 8,231,196,672 bytes (7,849.881 MiB). CUDA graphs and one-time selected-layer membership precomputation were enabled. The feature remains environment-gated and disabled by default.

## Performance

Ten alternating cold-process pairs generated 512 tokens each with the same Q6 model, prompt, context 2048, seed 42, 12 threads, `-ngl 99`, `--cpu-moe`, and CUDA graphs enabled on both sides. Every ordinary run was retained.

- Stock: 22.28 mean, 22.50 median, 0.992 sample SD decode TPS.
- ExpertFlow: 28.13 mean, 28.15 median, 0.211 sample SD decode TPS.
- Paired improvement: +26.47%; paired bootstrap 95% interval `[+23.46%, +29.83%]`.
- Improvement over the separately frozen strongest stock result of 22.967 TPS: +22.48%.
- Prompt TPS: 17.11 stock versus 22.23 ExpertFlow.
- Peak process-owned VRAM: 10,966.801 MiB, leaving 5,344.199 MiB of the reported 16,311 MiB total.
- Each mode produced one stable response hash across all ten runs. Cross-backend bit identity was not required by the protocol.

TTFT and token-latency percentiles were not collected after the terminal quality gate failed. Startup shadow copies also make total wall time unsuitable as a decode-latency proxy.

## Quality stop

The frozen held-out PPL protocol scored 8,184 paired tokens across eight context-2048 chunks. Stock PPL was 15,310.277 and ExpertFlow PPL was 14,863.951, a favorable point change of -2.92%. However, the paired block-bootstrap 95% interval was `[-8.03%, +2.25%]`; its upper bound exceeds the mandatory +1.0% limit. This gate therefore fails even though the point estimate improves.

The sprint stopped immediately. MMLU was not run because it cannot reverse the terminal PPL failure, and Stage 9 product commands were not implemented because packaging was authorized only after all final gates passed.

## Verification

The clean-checkout ExpertFlow suite passed with `216 passed, 6 skipped`; the four applicable static-placement source-contract tests passed against the isolated llama worktree. Enabling every historical source-contract test against this worktree is not a valid aggregate check: seven T1/T2 tests correctly reject the absence of temporal-cache APIs from this intentionally static branch.

## Reproduction and raw evidence

Commands are recorded in `reproduction-commands.md`. Raw performance artifacts are under `C:\models\expertflow\runs\q6-placement-final\stage8-authoritative\matched-10x512`; raw quality artifacts are under `C:\models\expertflow\runs\q6-placement-final\stage8-quality`. Hashes are recorded in `results.json`.
