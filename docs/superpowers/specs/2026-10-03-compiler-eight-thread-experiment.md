# Eight-thread stock-runtime experiment

Prerequisite: the committed fixed-budget A/A experiment at source commit 1192856
completed twenty runs with PASS-MEASUREMENT. Its direct mean was 22.974711 TPS,
sealed mean 23.068580 TPS, geometric change +0.400908%, and paired 90% interval
[-0.448648%, +1.337874%]. This is repeatability evidence, not an optimization.

## Hypothesis and numerical mechanism

The Ryzen 7 9700X has eight physical cores. The A/A measurements averaged 7.497
busy owned CPU cores and 16.864% GPU utilization. Reducing both native decode
and batch threads from twelve to eight may reduce scheduling contention during
CPU expert execution. Test only this change; do not sweep thread counts.

The pinned fork's ggml/src/ggml-cpu directory has no diff from upstream
a7312ae94f801fc9c6786dc56e38df57b964f697. The source revision is
451224ab4d12a616dc3e16e8c8063f4b331f531c. Q6_K repacking requires NEON in
repack.cpp:4655-4666, so the x86 Ryzen uses the standard expert dot-product path.
ggml-cpu.c:1463-1521 calls vec_dot for the complete reduction dimension of each
output; 1653-1704 changes ownership of output chunks, not the dot-product order.
Input quantization at 1599-1620 partitions complete Q8_K blocks; its block-local
conversion does not combine thread-local reductions. The row-based SwiGLU and
scale kernels in ops.cpp:3182-3239 and 4505-4554 preserve each row's arithmetic.
The binary, SIMD selection, weights, precision, GPU placement and CUDA graph
settings remain fixed. This supports scheduling equivalence for these inspected
kernels; it is not a universal bitwise guarantee for arbitrary CPU operations,
models or builds. Native prompt/generated token identity remains mandatory.

## Frozen protocol

Use the original pinned model/runtime/hardware, ngl99 CPU-MoE, F16 KV, graphs on,
context4096, predict512, seed42 and prompt. Threads are part of WorkloadIR and
therefore each arm has a different, explicitly recorded workload/candidate
identity. All fields except threads must match. Compare twelve versus eight
threads using ten balanced pairs, five of each order, shuffled with seed20261003.
Exactly twenty retained cold native processes; no retries or discarded outliers.
Any environment, identity, memory, token or cleanup failure stops immediately.

Use the same paired log-ratio bootstrap: 10,000 whole-pair draws, seed20261003.
PASS-OPTIMIZATION requires point gain >=5%, two-sided95% lower bound >0, both
arm CVs <=10%, exact tokens against the accepted A/A reference, and all mandatory
owned memory/cleanup checks. Otherwise declare INCONCLUSIVE (or VALIDATION-STOP
for an established regression). This small-sample estimate retains the temporal
dependence limitation. Never reinterpret A/A gains as optimization evidence.

Freeze this protocol, source files, source commit, prerequisite report/database,
identities, candidates and schedule before launching. Persist rows after each
run. Verify the prerequisite database remains unchanged. Report diagnostics
separately from native TPS. No automatic product-plan publication: an accepted
thread setting is evidence for a later explicit compiler workload-selection
design. No other hypothesis or architecture work is authorized in this gate.
