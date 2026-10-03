# Explicit phase-aware profiling result

**PASS-PHASE-PROFILING:** the separately pinned diagnostic build completed the
fixed two-control/three-profile budget on ef-v2. All five independent owned runs
passed native token, memory/reserve and cleanup checks. Every profile accounted
for 39 prompt tokens, 511 decode-forward tokens and all 62 decode splits. The
512-token response starts by sampling prompt logits, hence 511 decode forwards.
The two prompt graph calls correctly include a one-token prompt tail as prefill.

## Decode-only synchronized breakdown

| Run | Total split time | CPU compute | Input boundary | CUDA completion wait |
| --- | ---: | ---: | ---: | ---: |
| 1 | 21.666 s | 14.872 s (68.64%) | 2.315 s | 4.318 s |
| 2 | 22.237 s | 15.448 s (69.47%) | 2.262 s | 4.366 s |
| 3 | 22.378 s | 15.495 s (69.24%) | 2.288 s | 4.431 s |

Mean CPU compute share is **69.12%**; virtually all CPU compute is in the thirty
expert regions. Mean input-boundary time is 2.288s, CUDA completion wait 4.372s,
CUDA host submission 0.162s, and CPU completion wait 0.000658s. Warmup and
initialization counters are separate and excluded from this breakdown.

These are instrumented split durations, not an uninstrumented throughput result.
Per-split synchronization perturbs overlap. Input boundaries combine routing,
copies and waits; **pure transfer duration remains unavailable**. This supports
CPU expert work as a target, not a claim that cache misses or PCIe bandwidth are
the bottleneck. No performance optimization was tested or accepted here.

The stock control produced 23.1700 TPS and the profiling-off diagnostic control
22.7506 TPS. Their purpose was token/memory/cleanup validation; a single run per
runtime does not establish a performance difference. Profile-run TPS is likewise
diagnostic only. All timings retain the twelve-thread frozen workload.

## Implementation and verification

- Native source: `92811cbc13ff93b66e2162ebdcfbf7f2e8d59f9e` in the separate
  `C:/models/expertflow/worktrees/llama-phase-profile-20261004` worktree.
- The separate host-only build is
  `C:/models/expertflow/builds/llama-phase-profile-20261004`. CPU and CUDA kernel
  DLLs exactly match the previously pinned fork hashes; neither kernel source
  directory changed. Scheduler, llama/common and server code carry the explicit
  phase markers. Original builds/source remain unchanged.
- [Runtime manifest](../../../configs/compiler/runtime-phase-profile.json)
  pins every implementation DLL and ordered patch; the new patch is
  `release/expertflow-build-week/patches/llama.cpp/0007-feat-phase-aware-diagnostic-profile.patch`.
- Phase marking uses the states of token owners in the current server batch.
  Actual microbatch token counts and graph calls are separate from split calls.
  Counters are bounded at five phases by 256 split slots. Default-off adds no
  profiler synchronization and collects no counters. Mixed phases fail the
  frozen concurrency-one analysis.
- A compiled CPU graph with an eight-row expert-shaped output passed arithmetic
  and explicit phase/call accounting, directly exercising the old shape-label
  failure. Default-off left the profile file unchanged and emitted no profiling
  enabled message. Applicable native source contracts passed eight tests.
- One fresh review found missing split coverage, incoherent empty-phase counts
  and out-of-batch slot classification. Reproducing tests failed first; the fixes
  add authoritative split counts, strict phase accounting and batch-owner markers.
  The final full CPU result and log hashes are in verification.json.

The profiler still writes at graceful teardown; the tested private-console helper
verified owned PID/OS creation time and allowed all three native profile servers
to exit with code zero. No foreign process was stopped and no visible window was
opened. Profiling controls exist only in the isolated diagnostic probe; all
measurement rows have `measured=False` and cannot establish a product plan.

## Evidence and next hypothesis

[Analysis](analysis.json), [controls](controls-report.json),
[profiles](profiles-report.json), [raw split output](split-profile-1.json),
[artifact manifest](measurement-manifest.json) and [verification](verification.json)
retain the complete accounting and hashes. Raw files remain under
`C:/models/expertflow/runs/compiler-phase-profile-20261004`. The generic probe's
terminal string `PROFILED-AGGREGATE` is retained unchanged in its raw report;
its schema2 phase analyses and independently recomputed result establish the
explicit decode breakdown above. `probe/` preserves the throwaway launch code.

The next single proposal is a **one-row CPU expert-weight prefetch hint** before
the existing Q6 dot-product call. It targets the measured CPU region while
preserving arithmetic and reductions. Cache-stall evidence was not collected,
so benefit is uncertain and the hint may be neutral or slower. Its precise
source boundary, numerical/disassembly checks, kernel-only stock runtime,
fixed twenty-process paired budget and >=5% acceptance gate are frozen in the
[follow-up protocol](../../superpowers/specs/2026-10-04-cpu-expert-prefetch-experiment.md).
That experiment has not been executed.

The original Phase3 validation stop, eight-thread rejection and reactive-cache
no-go remain unchanged. No product gate was silently replaced, no plan was
published, and no merge or push was performed.
