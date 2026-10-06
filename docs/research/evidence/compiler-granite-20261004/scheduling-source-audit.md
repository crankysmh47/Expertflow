# Granite3.1 MoE Q6 source scope

Actual artifact is1,099,212,096bytes, SHA4566cfa92be10888026bd3663c83d64e91cd91f874dfb3607596587ff1c8f67f.
Publisher revision940d2e1f9f65330615c7c8e980e6c5ac73d3360c is immutable; the complete
inventory and official config are saved alongside this audit. The model has
24routed layers,32experts/top8, hidden1024/intermediate512, three Q6_K expert banks
per layer. Down has shape[512,1024,32]; gate/up[1024,512,32]. Q6 blocks contain
256elements/210bytes; each bank13,762,560bytes, layer41,287,680bytes and complete
expert bundle1,290,240bytes. The normalized inventory SHA is3de0a37570c518db90d027d256d2f5785d300582f5d68c2c5cfb6acbc132e569.

Pinned upstream a7312ae94f801fc9c6786dc56e38df57b964f697 explicitly supports
`granitemoe`. `models/granite-moe.cpp` loads router/gate/up/down expert tensors,
RMS norm, embedding/residual/attention/logit scales and rotary parameters.
`models.h` aliases its graph to Granite's graph; `granite.cpp` uses norm plus
softmax top-k routing, separate gate/up/down matrices, SILU and normalized expert
weights. Actual GGUF has no shared expert branch. This is a real different
architecture, not synthetic adapter data or another Gemma quantization.

The reviewed baseline places all24blocks and output on the existing GPU with
ngl99/CPU-MoE false. Own native reference stability and memory/reserve remain
mandatory. This baseline selection is based on the1.10GB footprint fitting the
16GB device, not a measured comparison against CPU-MoE or a global optimum.
The legacy input loader also verifies its configured distinct fork but no fork
binary or static/profile/cache path is executed in this track.

Threads alter host/CPU work partition; fixed GPU-resident expert arithmetic and
kernel arguments remain the same. CPU get-rows dequantizes complete independent
rows. CUDA graph mode changes capture/replay and kernel parameter submission,
with the same operations/kernels. The additional graph allocation/stream
optimizer remains disabled; inherited GGML/LLAMA/ExpertFlow controls are stripped.
The provider admits only threads and graph mode at this unchanged placement,
exact F16KV and own frozen semantic workload. Native prompt/generated tokens
must match this model's own reference; no Gemma token/speed/quality transfer.

The provider binds eighteen immutable source objects spanning Granite model and
graph construction, loader/architecture registration, CPU operators/dequantization,
CUDA dispatch/launch, quantization, expert MMQ/MMVQ, vector dot, get-rows, norm and
softmax. Exact object IDs are in [scheduling-source-audit.json](scheduling-source-audit.json).
The full pristine manifest identity326daa6e17e1293f6b7c5c9f23e85868961241f4dd00a94a3eee54e25071f629
and every actual server/DLL/CUDA runtime binding are independently checked.

Unsupported: other Granite weights/quantizations/inventories, changed builds or
sources, ARM/nonreviewed host architecture, partial affinity/topology, KV precision
changes, static placement, CPU/GPU placement comparisons, PDL controls, kernel
autotuning, flash-attention changes, shared-expert/hybrid variants or other backends.
GPU reference/product/search receipts must keep their declared provider baseline.
The audit is eligibility evidence; live performance and product acceptance are
separate gated steps.
