# Stage 1 narrow Q4 placement result

Status: **STOP**

## Passing physical proof

Implementation 1 used the pristine pinned llama.cpp source at `a7312ae94f801fc9c6786dc56e38df57b964f697`. It assigned only layer 0's routed gate/up, activation, and down operations to CUDA before scheduler splitting. Router/top-k, routing-weight application, aggregation, and residual execution remained on their original CPU plan at `-ngl 10`.

The scheduler created persistent, preallocated CUDA shadow operands and loaded the complete Q4 routed bundle once:

| Component | Packed bytes |
|---|---:|
| `blk.0.ffn_gate_up_exps.weight` | 285,474,816 |
| `blk.0.ffn_down_exps.weight` | 142,737,408 |
| `blk.0.ffn_down_exps.scale` | 512 |
| **Total** | **428,212,736 (408.375 MiB)** |

The original expert tensors remained CPU-backed. Scheduler diagnostics showed CPU router execution, a CUDA expert island, and an explicit return to CPU for `ffn_moe_weighted`/`ffn_moe_out`. The stable graph had five splits and no hidden full-model CUDA expert duplicate.

An early diagnostic incorrectly forced the scale tensor's view/reshape chain to CUDA and produced a reproducible illegal access. The bounded correction left dependency placement to the scheduler while keeping the scale shadow as the CUDA split input. The corrected 128-expert static shadow completed repeatedly. The failed commands and corrected result remain in the external raw-log directory.

## Exactness failure

The pristine reference, modified binary with the feature disabled, and enabled island each ran the same greedy 16-token probe at `-ngl 10`, one-token microbatches, and 12 threads. Each mode emitted 960 complete routing events (32 forwards × 30 MoE layers). Three disabled and three enabled repetitions were run.

- Modified-disabled tokens and router events matched pristine exactly in every repetition.
- Enabled runs were internally deterministic across all three repetitions.
- Enabled prompt tokens matched pristine.
- Enabled generated tokens did **not** match pristine. The first difference was generated index 1: reference `236770`, candidate `236796`.
- The first router difference occurred during prefill token 0 at layer 13. The reference ended its top eight with `[23, 127]`; the CUDA-island run used `[127, 23]` in those final positions.
- Both traces remained complete at 960 events; this is numerical ordering drift, not missing telemetry.

The reproducible analyzer is `scripts/compare_narrow_placement.py`. Raw inputs are under `C:\models\expertflow\runs\q6-runtime-final\stage1\layer0-parity`.

## Second-design disposition

The audited public scheduler surface supports pre-scheduling backend assignment and then performs copy insertion/allocation internally. It does not expose a graph-builder copy operation capable of changing the CUDA expert kernel's numerical result. An explicitly constructed second boundary would therefore execute the same `MUL_MAT_ID` CUDA kernels and retain the measured drift. Restoring CPU-exact results would require a replacement numerically matching kernel, retaining CPU expert computation, a new execution operation, or broad scheduler replacement. All are declared stop conditions.

Implementation 2 was therefore rejected before source modification. Layer 15, the naturally CUDA-resident control, reactive caching, Q6 generalization, and product benchmarks were not started because layer 0 failed the required validation order.

The abandoned llama.cpp source patch was not committed. Its final audit diff hash was `2b0038f553add4d8a59b9acc094e9b94a9d014bd`; the diagnostic binary SHA-256 was `36b083b6091c3079f8d8e45208aba629073d60db6582bd547e2ff8fab79b4c40`.

**Verdict: NARROW PLACEMENT STOP**
