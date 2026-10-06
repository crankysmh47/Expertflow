# Gemma Q6 CUDA scheduling: no-go, accepted incumbent retained

The registered twenty-process experiment completed once at source
`34a19b29dfc1846d86a0ddda5d2411deba7e3fd6`. Ten balanced pairs compared default
PDL enabled against explicit `GGML_CUDA_PDL=0`, with all other model, workload,
runtime, placement, threads, CUDA graph, KV and host controls fixed.

| Arm | Mean decode TPS | Sample CV |
| --- | ---: | ---: |
| Default PDL enabled | 22.2158673 | 1.9597% |
| PDL disabled | 22.0025213 | 1.6984% |

Disabling PDL changed paired geometric TPS by **-0.9562%**, with percentile
bootstrap CI95 **[-1.6383%,-0.2369%]**. It fails the frozen gain>=2%/CI lower>0
gate. All twenty native processes passed own Q6 reference tokens, ownership,
memory/reserve and cleanup. There were no retries, excluded samples, altered
gates or competing measurements. The lower absolute TPS than earlier experiments
is descriptive; only this fresh paired contrast supports the scheduling verdict.

The collector's separate read-only CLI audit passed. An independent verifier
reconstructed native completion timings, controls, twenty unique owners, tokens,
sample CVs and bootstrap statistics; all28 source files matched their actual Git
revision. See [verification.json](verification.json), [report.json](report.json)
and [frozen-protocol.json](frozen-protocol.json). Raw native artifacts and database
remain at `C:/models/expertflow/runs/compiler-cuda-pdl-20261004`; measured source is
preserved at `C:/models/expertflow/worktrees/compiler-cuda-pdl-34a19b2`.

Accepted Q6 plan
`3849427ac69fdab14babedb00d0a3b3fd4c0a44420a7b8aa32963fe805cc3497`
is unchanged: twelve threads, CUDA graphs on, CPU-MoE, pristine runtime, default
PDL enabled. Q4 remains separately validated at its own quantization and plan.
Original static quality-stop, reactive-cache/mover no-go, thread-search results,
CPU-prefetch inconclusive result and historical replay validation-stop stand.

This closes the authorized bounded Gemma optimization work; it does not prove a
global optimum or exclude future separately justified kernel research. The next
work is real second-family validation, with a separate artifact/reference and
source eligibility rather than transferring Gemma's speed or placement claims.
