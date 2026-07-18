# Stage 0 truthful Q6 baseline audit

## Frozen result

The fair stock baseline is pristine pinned llama.cpp `a7312ae94`, built with MSVC 19.39/CUDA 12.8, at `-ngl 99 --cpu-moe`. Three cold processes generated 512 tokens each at 23.3, 22.8, and 22.8 decode TPS (mean 22.967 TPS), with an identical normalized response hash. Peak process-owned VRAM was 3,098.656 MiB and mean prompt-evaluation/TTFT was 651.493 ms. The 1.25x product target is therefore 28.708 decode TPS.

## What “31/31 CUDA layers” actually meant

It did not prove full expert residency. With no tensor override, the matched pristine runtime reported 21,788.44 MiB of CUDA model allocation intent, only a 577.50 MiB CPU token embedding, two scheduler splits, 14,588.004 MiB physical process-owned peak, and 6.4 decode TPS. Windows/CUDA can expose allocation intent larger than physically resident VRAM; process-owned peak is kept separate.

Stock `--cpu-moe` still reports 31/31 layers offloaded, but the reconciled allocation is different: 21,783.63 MiB CPU-mapped model source, 2,163.72 MiB CUDA non-expert allocation, 62 graph splits, and 3,098.656 MiB physical process-owned peak. The router and surrounding dense work stay CUDA-planned; routed `MUL_MAT_ID` operations follow their CPU expert operands. This is genuine upstream stock behavior and is dramatically faster on this machine.

## Disabled-mode audit

The patched b910bc37 cache source differs from a7312ae94 only in the ExpertFlow cache/backend files. With every `EXPERTFLOW_*` variable cleared, the matched MSVC pristine and patched-disabled full-CUDA-intent runs produced the same 512-token normalized hash `2aa83308...4fe22`, the same allocation/split topology, and 6.4 decode TPS. Patched-disabled `--cpu-moe` also matched the pristine CPU-MoE response hash `ebaed9f6...28f14`. No disabled-mode fix was required.

The packaged official binary was retained only as a diagnostic because it was built with Clang 20 rather than MSVC and produced a different deterministic stream. It is not used to judge disabled equivalence or set the product target.

## Tensor and operation placement

`placement-manifest.json` lists every GGUF tensor for full-CUDA-intent and CPU-MoE modes. Raw tensor bytes reconcile with the verbose runtime buffers once tied output allocation, alignment, and mmap source mapping are distinguished:

- Full-CUDA-intent: routed experts CUDA; router, expert matmuls, and dense surroundings CUDA; token embedding source CPU-mapped/tied output CUDA allocation.
- CPU-MoE: routed experts CPU-mapped; routed expert matmuls CPU; router and dense surroundings CUDA; ordinary scheduler boundaries produce 62 graph splits.
- There is no ExpertFlow cache or predictor in either stock mode.

Raw stdout/stderr and process-owned VRAM samples are under `C:\models\expertflow\runs\q6-runtime-final\stage0`.
