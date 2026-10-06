# Bounded stock repeatability and workload transfer

Registered October 4; collected October 5, 2026 locally on `ef-v2`, source
`86388e7529b1fa8fa927e2baebc3f5ce84c6d597`.

Terminal status: **PASS-STOCK-REPEATABILITY-TRANSFER**, process exit 0. All148
native calls completed, both consumers passed, and the frozen collector's complete
source/phase/receipt reconstruction passed before persisting PASS. The
[independent final raw audit](final-audit.json) reconstructed all native results
and binds the final report SHA. Fresh read-only CLI validation also passed with
exit 0, launching no model. See [native verification](native-verification.json),
[validation log](final-validation.log) and [collector report](collector-report.json).

The [protocol](../../superpowers/specs/2026-10-04-stock-repeatability.md) retained
the existing gates, waited at least 30 seconds before every native call, and
recorded optional diagnostics. Main blocks were judged independently, without
pooling. The untouched code workload received the unchanged stock-tuning method
only after both main blocks and the fresh main consumer passed.

## Native gates

| Gate | Retained calls | Result |
| --- | ---: | --- |
| Main repeatability A | 20 | PASS; paired change -0.0825%, CI90 [-0.5529%, +0.5396%] |
| Main repeatability B | 20 | PASS independently; +0.2741%, CI90 [-0.6408%, +1.4329%] |
| Fresh main consumer | 1 | MEASURED-ACCEPTED-STOCK |
| Transfer reference | 10 | PASS; CV 4.7135%, all ten samples retained |
| Transfer automatic/manual grids | 18 + 18 | Each independently selected 12 threads, CUDA graphs on |
| Transfer defaults comparison | 20 | +9.3756% geometric gain, CI95 [+7.9238%, +10.6130%] |
| Transfer manual comparison | 20 | Equivalent; +0.1007%, CI90 [-0.3890%, +0.6453%] |
| Transfer fresh product | 20 | PASS-STOCK-FALLBACK; +0.3676%, CI90 [-0.2838%, +1.4386%] |
| Fresh transfer consumer | 1 | MEASURED-ACCEPTED-STOCK |

The transfer defaults arms averaged 22.0034/24.0584 decode TPS. The gain above
is the paired geometric estimate, rather than a ratio of those arithmetic means.
All four utility arm CVs were at most 3.0265%; product CVs were 0.1743%/1.6639%.
The original +/-2% equivalence, >-2% one-sided noninferiority, <=10% variance,
exact-token, identity, owned-memory, reserve and cleanup gates were retained.

The six configurations were 8/12/16 threads with graphs on/off, three complete
blocks per search. Defaults mean resolved 8 threads/graphs on, with CPU-MoE,
placement, workload and F16 controls fixed. This does not test every out-of-box
runtime setting. There were no extra candidates, retries or discarded samples.

## Costs and diagnostics

The registered collector interval was **11,717.250 seconds** (195.2875 minutes).
It starts after input loading and ends after collection/publication, before
read-only reconstruction. It is not total CLI wall time or deployment throughput.

| Cost | Seconds |
| --- | ---: |
| All fixed prelaunch waits | 4,441.791; minimum individual wait 30.000 |
| Native load/tokenize/completion/teardown phases | 5,364.040 |
| Automatic search native phases, 18 evaluations | 685.212 |
| Manual search native phases, 18 evaluations | 667.924 |

Equal search cost refers to native evaluation counts. Automatic tuning did not
have lower measured native phase time. Waiting, source/host checks, metadata,
plan preparation and read-only validation must not be hidden in decode TPS.
The frozen final validator creates a fresh reader for each attempt and rehashes
the 22.9 GB model repeatedly; its substantial post-collection time is outside
the registered collector interval. Reader reuse in the independent auditor uses
the EvidenceStore model cache and preserves per-record artifact verification.
The separate fresh read-only CLI validation took **2,191.484 seconds**
(36.5247 minutes), with zero additional native calls. This validation cost is
additional to the registered collector interval.

Peak process-owned GPU memory was 3,136.660 MiB; minimum observed device free
memory was 10,975 MiB. Measurement diagnostics retained 21,434 samples each for
GPU clocks, temperature and power, CPU reported frequency and system utilization;
CPU performance had 21,425 available samples. GPU temperature ranged 39-49 C.
CPU temperature is unavailable. These observations do not establish the cause of
the original inconclusive study or prove that waiting repaired thermal behavior.

## Scope and next work

This study tests stock scheduling on the pinned Gemma 4 26B A4B Q6_K artifact,
two frozen workloads, one Windows/RTX 5060 Ti host and pristine runtime build.
It qualifies the bounded stock-autotuning method under this timing/diagnostics contract.
It does not demonstrate custom placement acceleration, a global stock optimum,
superiority to manual tuning, universal gains, or steady-state serving performance.
Q4 and Granite's earlier stock acceptance establish separate compatibility,
rather than defaults-tuning utility on those artifacts.

The [original utility study](../stock-utility-20261004/report.md) remains closed
at PRODUCT-VALIDATION-STOP, 106/107 calls: +12.70% main defaults gain but
inconclusive product equivalence. New acceptance does not rewrite that verdict.
Historical placement QUALITY STOP and cache/thread/prefetch/PDL no-go results
also remain closed.

Next product work is public CLI consolidation, clear coverage/cost/invalidation
reports, and separately registered wider workload/host utility validation.
Preserve measured source snapshots when changing source-bound validation code.
Address reader reuse before making interactive validation performance claims.

Raw authority is
`C:/models/expertflow/runs/compiler-stock-repeatability-20261004`.
Main acceptance is `block-b/accepted`; held-out acceptance is
`transfer/utility/product/accepted`, each with execution-plan.json and
acceptance-receipt.json. Native databases and per-call artifacts remain there.
See [execution state](execution-state.md), [implementation review](implementation-review.md),
[implementation verification](implementation-verification.json), and
[independent audit controls](audit_controls.py). Archived phase authorities are
[main A](block-a-report.json), [main B](block-b-report.json),
[transfer utility](transfer-report.json), and [transfer product](transfer-product-report.json).
The [source integrity check](source-integrity.json) preserves 41 frozen files,
six historical pins and the original closed report. Per-call artifacts remain
in the native root; these snapshots do not relocate their identity-bound paths.
