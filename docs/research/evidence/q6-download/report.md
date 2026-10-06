# Q6 download and stock characterization

## Result

The requested Q6_K GGUF is verified at `C:\models\gemma-4-26b-a4b-q6\google_gemma-4-26B-A4B-it-Q6_K.gguf`: 22,862,575,520 bytes, SHA-256 `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.

The bounded stock runtime reached the maximum meaningful setting, `-ngl 99`, and reported all 31/31 layers on CUDA. Three post-discovery repetitions averaged 8.833 prompt TPS and 4.567 decode TPS; maximum process-owned VRAM was 14,592.051 MiB. The initial discovery run was materially colder/slower (3.4 prompt TPS, 1.3 decode TPS) and is preserved separately rather than folded into the measured repetition summary.

## Measured facts

- C: free space: 93,687,873,536 bytes before; 65,190,268,928 bytes after.
- Model: Q6_K, repository revision `fabed3e586120477355eea23b92644540a79ce2f`.
- Inventory: 658 tensors and 30 routed MoE layers.
- Every routed layer has the same three expert-indexed components: `ffn_gate_up_exps.weight`, `ffn_down_exps.weight`, and `ffn_down_exps.scale`.
- Exact packed bundle transfer size: 5,358,852 bytes per logical expert per layer.
- Total routed-expert bank: 20,577,991,680 bytes; non-expert tensor payload: 2,268,760,696 bytes.
- Stock runtime: llama.cpp b10002 at upstream `a7312ae94`, ExpertFlow disabled, RTX 5060 Ti, driver 591.86.
- Stable stock placement: `-ngl 99`, 29 repeating layers plus output, 31/31 offloaded.
- Repetition decode TPS: 4.5, 4.5, 4.7. Repetition prompt TPS: 8.9, 8.9, 8.7.
- Process-owned peak VRAM: 14,592.051 MiB in each measured repetition.

## Calculated memory values

The metadata lower-bound dense/non-expert GPU floor is 2,373,114,980 bytes: non-expert tensor payload plus the verbose run's 99.52 MiB CUDA compute buffer. With a 16,310 MiB runtime-reported device budget and a 256 MiB margin, the largest arena in the requested list that fits this lower-bound equation is 88 slots across all 30 routed layers (14,147,369,280 arena bytes).

| Slots/layer | Arena bytes | Freed vs full 128 bank | Lower-bound fit |
|---:|---:|---:|:---:|
| 128 | 20,577,991,680 | 0 | no |
| 112 | 18,005,742,720 | 2,572,248,960 | no |
| 96 | 15,433,493,760 | 5,144,497,920 | no |
| 88 | 14,147,369,280 | 6,430,622,400 | yes |
| 80 | 12,861,244,800 | 7,716,746,880 | yes |
| 72 | 11,575,120,320 | 9,002,871,360 | yes |
| 64 | 10,288,995,840 | 10,288,995,840 | yes |
| 56 | 9,002,871,360 | 11,575,120,320 | yes |
| 48 | 7,716,746,880 | 12,861,244,800 | yes |
| 32 | 5,144,497,920 | 15,433,493,760 | yes |

## Projections

The 88-slot result is only a theoretical metadata-floor capacity. It assumes the full 128-expert bank is replaced rather than duplicated and does not establish an executable cache. The reported freed bytes compare packed GGUF expert-bank payloads; they are not measured physical VRAM savings from a live cache.

## Unresolved architecture blockers

The prior Stage 1 `PLACEMENT STOP` remains in force. No scheduler, graph placement, cache, predictor, or ExpertFlow runtime source was modified here. A live implementation would still need to prove target-layer CUDA placement before scheduling, eliminate hidden full-bank CUDA duplicates, account for staging/mappings/fragmentation, and pass exactness and cleanup gates. Therefore no Q6 cache feasibility or speedup claim is made.

Raw logs are external at `C:\models\expertflow\runs\q6-download`; the repository contains the verified manifest, complete tensor inventory, summarized stock evidence, and inventory tooling/tests only. The 22.9 GB model is not tracked.
