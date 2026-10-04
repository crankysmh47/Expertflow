# Placement feasibility audit

Verdict: **NO-GO within the current exact numerical contract**; activate the
conditional stock-utility proof. No new native inference was run for this audit.
This is a scoped feasibility decision, not a claim that all MoE placement or
quality-bounded acceleration is impossible.

## Numerical mechanism

The pinned scheduler duplicates packed tensor layouts into persistent CUDA
shadows and copies the original bytes once. No repacking defect was found in
that code boundary. The physical bundle is complete and identity-mapped.
Nevertheless, identical weight bytes do not imply identical operations:

- CPU Q6_K uses Q8_K activation quantization and CPU dot/reduction routines
  (`ggml-cpu.c` type traits and `ops.cpp`).
- CUDA routed matmuls dispatch to MMVQ/MMQ/MMF (`ggml-cuda.cu`); the quantized
  vector path uses Q8_1 activation blocks and CUDA dot/reduction/fusion routines
  (`mmvq.cu`, `vecdotq.cuh`).
- The earlier narrow-placement trace already measured deterministic token and
  router-order divergence. Changing scheduler boundaries or copying the same
  weights cannot make those arithmetic paths identical.

Restoring CPU-exact arithmetic on CUDA would require a new matching kernel and
its own numerical/performance project. Keeping CPU execution preserves the
current contract but provides no new placement-based CPU compute relief. The
current compiler correctly rejects these candidates as `numerical_path_change`.

## Historical quality reconstruction

Recomputed the paired NLL analysis from all 8,184 original records, 64 blocks,
10,000 bootstrap draws and seed20260719. PPL point change is -2.9152085407%;
the upper 95% bound is +2.2529372656%, exceeding the original +1% gate. The
reconstruction matches the recorded result. This is a failure to establish the
required quality bound, not evidence that quality certainly worsened or a
diagnosis of tensor corruption. Absolute PPL is high in both arms; its cause is
unresolved and was not changed or used to waive the paired gate.

## Alternative reviewed

Pinned upstream `--n-cpu-moe` and `--override-tensor` already support partial
expert tensor residency. The loader uses the first matching buffer override.
This can avoid the fork's persistent shadow duplication, but residency still
selects CUDA arithmetic. It therefore has no source-backed exactness advantage
over the rejected path. It belongs to a future separately quality-qualified
stock-placement space, and is not a new custom runtime speedup against strongest
stock. No new quality defect/mechanism was established that would justify
repeating the old failed static hypothesis or extending its sample budget.

## Decision and limits

No eligible new exact placement hypothesis was found within the existing
operations and bounded plan. Leave placement proof/transfer stages conditional;
do not launch an unchanged approximate placement experiment merely to obtain a
passing interval. Proceed with pristine-runtime thread/graph autotuning, whose
reviewed operation paths and native guards are already implemented.

[`audit.json`](audit.json) pins nine inspected source files, the native revision,
both original NLL files and the reconstructed statistics. Historical verdicts
and accepted Q6/Q4/Granite plans remain unchanged. Relevant original evidence:
[narrow placement](../q6-runtime-final/stage1-narrow-placement-stop.md),
[quality stop](../q6-placement-final/report.md), and
[phase profiling](../compiler-phase-profile-20261004/report.md).
