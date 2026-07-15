# Q4 minimal live-cache bounded-spike decision

> **Superseded by tracing quarantine:** The empirical routing inputs to this decision are labeled `trace_v1_perturbing` and excluded from final claims. Gate 4 is closed and this historical conditional-go is not actionable until a parity-safe replacement corpus is collected and re-evaluated. See `configs/trace-evidence-status.json`.

Date: 2026-07-15 PKT

Historical decision: **CONDITIONAL-GO-FOR-BOUNDED-SPIKE; NOW SUSPENDED.** The protected Observatory remains the submission floor. Do not proceed to a cache proof while the observer gate is open.

This decision permits a correctness experiment, not a production cache or runtime-speedup claim. The projected arena fits and idle CUDA transfer measurements justify checking the physical path. The static policy result is positive on a genuinely held-out split, but its margin over LRU is small enough that expansion still needs a stronger same-runtime result.

## Gate review

| Gate | Result | Evidence |
| --- | --- | --- |
| Cache physically fits | Pass as a projection | 96 slots in each of 21 layers produces 2,016 slots. At 3,346,048 bytes per slot, the arena is exactly 6,745,632,768 bytes / 6,433.136719 MiB. This leaves 800.863 MiB of the measured configurable envelope. No live allocation has been attempted. |
| Transfer timing leaves meaningful headroom | Inconclusive | Three idle-CUDA trials put an aligned pinned slot at 0.234016 ms p50 / 0.234272 ms p95. The p95 two-slice value is 0.236288 ms. A cross-backend perfect-future simulation reduces transfer-only residual from 4.9335 to 0.1705 ms/token, but CUDA copy/compute contention and in-model CUDA windows are unmeasured. |
| A practical non-oracle policy remains promising | Conditional pass for a bounded spike | Static-96 was selected exclusively from 31 parity-safe training conversations, then scored on eight untouched conversations: four validation and four test. It reaches 87.57% versus 86.34% for conversation-reset LRU and removes 9.03% of LRU's cold bytes. That is enough to test the exact physical path, but it remains below the earlier 20% expansion target and loses on code, structured output, and topic shift. |
| llama.cpp change can stay isolated and off by default | Not demonstrated | The existing router probe is a separate executable and proves that telemetry can stay outside the official runtime path. A cache could be designed behind an off-by-default flag, but no cache allocation, replacement, or parity-preserving feature-flag boundary exists yet. |

The held-out methodology correction changes the verdict: static-96 is not an in-sample oracle, so a minimal exact blocking proof is now justified. The two unmeasured runtime boundaries still prevent any expansion or performance claim until direct evidence closes them.

## Held-out split verification

The serialized report lists 31 `train-*` fit traces. The evaluation set contains only `validation-*` and `test-*` conversations, one from each of the eight domains. In `build_held_out_capacity_curve`, resident counts come only from `training_events`; hit and miss totals come only from `evaluation_events`. The corrected result therefore has conversation-level separation for both static selection and scoring.

This verification changes the eligibility verdict, not the measured effect size. Static-96 still cuts cold bytes by 9.0257% rather than the earlier 20% target.

## Evidence used

- Collection manifest: 40 conversations, 39 exact-parity pairs, one explicitly excluded training failure; SHA-256 `47e2154870c50b2aed9aef148a7b9a6496173d2a1516529c744fa7f5a1981093`
- Corrected held-out capacity curve: conversation-reset LRU, 31 training conversations and 8 held-out conversations; SHA-256 `05966196a75f4a0e77b61137e4ab8250cc4f78bc922ebbfcf9c4580d2fcaf678`
- Exact 3,840-object inventory; SHA-256 `daf9a54c1d03a933a667644de412038fb1530ee90ef1761f2f74dbdacb5f1b7a`
- Three-trial transfer aggregate; SHA-256 `fb90c8820085f80849977cbf7849de2c899d9c1a4dfd20b5e3fe20e63244b94b`
- P95 cross-backend deadline sensitivity; SHA-256 `4d46952c1eee603ce6bab09201a75c116f1b493aa061283ba8a9b296a320e8ee`
- Current recommendation: `CONDITIONAL`, static-96 replay, `live_cache_enabled=false`; SHA-256 `cb8c0d33e888a072646c0d0a47a4e5792686b3d717b563e36ca9ff3805955322`

CUDA microbenchmarks, Vulkan callback timing, simulator estimates, oracle results, and live runtime measurements remain separate evidence classes. There is no live-runtime measurement in this checkpoint.

## Bounded permission

First protect and reproduce the Observatory, install the supported VS 2022/v143 and CUDA 12.8 toolchain, and build exact llama.cpp commit `a7312ae94f801fc9c6786dc56e38df57b964f697` without live-cache changes. Stop if that clean build cannot reproduce known-good behavior.

If the clean gate passes, the first experiment is one layer, true-router-selected exact blocking loads, deterministic forced replacement, direct execution from preallocated Q4 slots, and parity. The flag is disabled by default. No asynchronous prefetch, learned predictor, MTP controller, KV-cache claim, full static-96 allocation, or speedup claim belongs in this spike.

`live_cache_enabled=false`
