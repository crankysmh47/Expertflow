# Q4 minimal live-cache go/no-go

Date: 2026-07-15 PKT

Decision: **NO-GO for the minimal live-cache spike on the current evidence.** Keep shipping the Observatory and simulator. Do not modify the live llama.cpp cache path yet.

This is a bounded decision, not a claim that expert caching cannot work. The packed arena fits on paper and the perfect-future simulation shows useful transfer headroom. The practical static policy is the failing gate.

## Gate review

| Gate | Result | Evidence |
| --- | --- | --- |
| Cache physically fits | Pass as a projection | 96 slots in each of 21 layers produces 2,016 slots. At 3,346,048 bytes per slot, the arena is exactly 6,745,632,768 bytes / 6,433.136719 MiB. This leaves 800.863 MiB of the measured configurable envelope. No live allocation has been attempted. |
| Transfer timing leaves meaningful headroom | Inconclusive | Three idle-CUDA trials put an aligned pinned slot at 0.234016 ms p50 / 0.234272 ms p95. The p95 two-slice value is 0.236288 ms. A cross-backend perfect-future simulation reduces transfer-only residual from 4.9335 to 0.1705 ms/token, but CUDA copy/compute contention and in-model CUDA windows are unmeasured. |
| A practical non-oracle policy remains promising | Fail | On eight held-out conversations, static-96 reaches 87.57% versus 86.34% for conversation-reset LRU. It removes only 9.03% of LRU's cold bytes, below the 20% gate, and loses on code, structured output, and topic shift. |
| llama.cpp change can stay isolated and off by default | Not demonstrated | The existing router probe is a separate executable and proves that telemetry can stay outside the official runtime path. A cache could be designed behind an off-by-default flag, but no cache allocation, replacement, or parity-preserving feature-flag boundary exists yet. |

Because all four gates must support proceeding, the practical-policy failure is enough to stop. The two unmeasured runtime boundaries make the same decision safer.

## Evidence used

- Collection manifest: 40 conversations, 39 exact-parity pairs, one explicitly excluded training failure; SHA-256 `47e2154870c50b2aed9aef148a7b9a6496173d2a1516529c744fa7f5a1981093`
- Corrected held-out capacity curve: conversation-reset LRU, 31 training conversations and 8 held-out conversations; SHA-256 `05966196a75f4a0e77b61137e4ab8250cc4f78bc922ebbfcf9c4580d2fcaf678`
- Exact 3,840-object inventory; SHA-256 `daf9a54c1d03a933a667644de412038fb1530ee90ef1761f2f74dbdacb5f1b7a`
- Three-trial transfer aggregate; SHA-256 `fb90c8820085f80849977cbf7849de2c899d9c1a4dfd20b5e3fe20e63244b94b`
- P95 cross-backend deadline sensitivity; SHA-256 `4d46952c1eee603ce6bab09201a75c116f1b493aa061283ba8a9b296a320e8ee`
- Current recommendation: `CONDITIONAL`, static-96 replay, `live_cache_enabled=false`; SHA-256 `cb8c0d33e888a072646c0d0a47a4e5792686b3d717b563e36ca9ff3805955322`

CUDA microbenchmarks, Vulkan callback timing, simulator estimates, oracle results, and live runtime measurements remain separate evidence classes. There is no live-runtime measurement in this checkpoint.

## What would change the decision

Reconsider the blocking exact-cache experiment only after a practical, training-only policy clears the declared improvement threshold on a larger held-out set without collapsing on topic shifts. Before touching the live path, also measure transfer under representative CUDA model contention and write down an off-by-default feature-flag boundary that can be removed cleanly.

If those gates pass, the first experiment is still intentionally small: exact blocking loads, deterministic slot replacement, and token parity. No asynchronous prefetch, learned predictor, MTP controller, KV-cache claim, or speedup claim belongs in that spike.

`live_cache_enabled=false`
