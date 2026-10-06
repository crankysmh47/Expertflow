# One CPU expert-row prefetch hypothesis

Status: frozen follow-up proposal, not executed in the phase-profiling task.
Phase-aware diagnostic results identify CPU compute as 68.64-69.47% of
synchronized decode split time; virtually all CPU compute is in expert regions.
Input boundaries and CUDA completion are secondary in that instrumented view.
This identifies a target, not a proof of cache misses or memory-bandwidth limits.

Hypothesis: a one-row lookahead cache hint before the existing Q6 expert
dot-product call can hide some expert-weight access latency without changing
the dot-product arithmetic. Choose exactly this candidate, not a distance sweep.
Software hints may add overhead or provide no benefit; those outcomes are valid.

## Candidate and numerical contract

Start from pristine upstream a7312ae94f801fc9c6786dc56e38df57b964f697 in a new
isolated native worktree. CPU source is unchanged between that base and the
existing ExpertFlow fork, but use the pristine host runtime for the comparison.
Change only the CPU expert MUL_MAT_ID caller in ggml/src/ggml-cpu/ggml-cpu.c,
inside ggml_compute_forward_mul_mat_id_one_chunk. For Q6_K only, before its
existing vec_dot call, hint the first cache line of the next output-weight row
using _mm_prefetch with _MM_HINT_T0, only when ir0+1 < ir0_end. The address must
remain inside the same expert allocation. No hint writes data or runs across an
expert/chunk boundary. Keep the complete current dot-product call and reduction
dimension, output stores, activation conversion and iteration order unchanged.

Rebuild only the CPU backend DLL with the original stock compiler/CPU flags.
Pin a separate kernel-only runtime manifest using pristine stock launchers,
scheduler, llama/server, CUDA and all other DLLs; replace only ggml-cpu.dll.
Do not use the instrumented phase-profile host binaries for acceptance timings.
Verify all untouched dependency hashes and imported symbol compatibility.

Before native runs, compare stock/candidate quantized CPU expert outputs bitwise
for deterministic Q6/Q8 fixtures, including chunk tails and expert boundaries.
Inspect the dot-product source and normalized disassembly: its arithmetic and
reduction instructions must remain unchanged. Compiler flags, SIMD selection
and floating-point options cannot change. Any discrepancy is an exactness stop,
not something finite native token parity can excuse. Default-off/profile-free
timing must contain no synchronized diagnostic instrumentation.

## Fixed acceptance experiment

Use the strongest twelve-thread pristine stock control and the kernel-only
candidate on the exact pinned context4096/predict512/seed42/F16/graphs-on/ngl99/
CPU-MoE workload and model. Fresh database/output; ten balanced pairs, twenty
retained cold native processes; seed20261003, five of each order. No retries,
discarded samples, tuning or alternative prefetch distances after results.

Use paired log-ratio percentile bootstrap, 10,000 whole-pair draws, seed20261003.
PASS requires >=5% point gain, two-sided95% lower bound >0, both CVs <=10%, exact
prompt/generated tokens against the reference and all owned memory/reserve,
process identity and cleanup checks. Preserve every rate and raw artifact hash.
Failure/inconclusive ends the hypothesis. No automatic compiler-plan publication.

This is an experiment to answer whether the hint helps, not an assertion that
the system is memory-bound. Do not reopen eight threads, reactive GPU caching,
CPU-to-CUDA numerical placement, SIMD retuning or additional hypotheses here.

Intrinsic semantics reference: Intel's
[cacheability support intrinsics](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/cacheability-support-intrinsics-002.html).
Processor-specific performance remains empirical and cannot be inferred from
the instruction's availability.
