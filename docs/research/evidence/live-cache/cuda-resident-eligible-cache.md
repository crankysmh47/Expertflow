# CUDA-resident eligible-layer cache result

**Verdict:** two-layer correctness PASS; all-eligible `-ngl 10` exactness PASS;
reactive performance FAIL. Stop before higher `-ngl`, 64 slots, or predictor
integration.

The current cache is now constrained by the measured eligibility rule:

> A layer may use the current ExpertFlow cache only when its relevant MoE
> execution path is already CUDA-resident under the selected `-ngl`
> configuration.

No graph relocation, whole-layer offload, hybrid CPU/GPU expert execution,
backend splitting, asynchronous copy, prediction, MTP, or kernel change was
added. The feature remains disabled by default.

## Eligibility behavior

Natural scheduler assignment runs before any cache-specific expert redirection.
For each requested layer, the runtime checks the same-layer MoE router backend,
the exact two `MUL_MAT_ID` consumers, CUDA operation support, exact tensor
identity/layout, and host-backed source tensors used by the proven cache path.

An explicit request fails if any requested layer is ineligible. Auto mode alone
may filter the request. The layer-0 `-ngl 10` rejection control exits before
arena allocation with:

```text
rejected layer=0 backend=CPU reason=moe_router_not_cuda
explicitly requested layer 0 is ineligible
```

At `-ngl 10`, auto discovery reports:

- requested layers: `0..29`;
- eligible CUDA layers: `21,22,23,24,25,26,27,28,29`;
- rejected CPU layers: `0..20`, each with `moe_router_not_cuda`;
- slots: 32 per eligible layer;
- planned arena: 963,661,824 bytes;
- exact allocated arena: 963,483,264 bytes (918.849 MiB).

The auto smoke completed one generated token, produced 18 ordered cache events
for nine layers, and cleaned up.

## Two-layer `[21,24]` ramp

Protocol: `-ngl 10`, 12 threads, fixed general/code/translation prompts, one
warmup plus three measured repetitions per mode and domain, 64 requested
generated tokens, full canonical routing trace, and 100 ms system GPU samples.

All nine measured cache-on repetitions match cache-off exactly for prompt
tokens, generated tokens, every ordered router selection, cache-to-router
selected IDs, event counts, and layer order. Independent event replay confirms
that mappings and generations remain isolated by layer with no stale or
cross-layer slot state.

- exact consolidated arena: 214,107,392 bytes;
- measured events: 1,830;
- demands: 14,640;
- hits: 10,749;
- misses: 3,891;
- aggregate hit rate: 73.42%;
- bytes transferred: 13,016,998,092;
- aggregate host-wall blocking time: 3,381.522 ms;
- blocking per generated token: 7.18 ms, including prefill transfers.

Per layer:

| Layer | Events | Hits | Misses | Hit rate | Bytes | Blocking ms |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 915 | 5,463 | 1,857 | 74.63% | 6,212,430,084 | 1,628.776 |
| 24 | 915 | 5,286 | 2,034 | 72.21% | 6,804,568,008 | 1,752.746 |

Focused means:

| Mode | Prompt TPS | Decode TPS | End-to-end ms | System GPU peak mean |
|---|---:|---:|---:|---:|
| cache off | 22.391 | 26.366 | 4,248.737 | 6,710 MiB |
| layers 21+24 | 20.079 | 26.255 | 4,506.012 | 6,238 MiB |

Relative to matched cache-off, prompt TPS is -10.32%, decode TPS is -0.42%,
and end-to-end time is +6.06%. Decode behavior is approximately neutral rather
than a severe regression. The end-to-end increase is bounded by the directly
measured blocking-transfer total.

## All eligible `-ngl 10` ramp

The same protocol was repeated for explicit layers `21..29`, after auto mode
discovered that exact eligible set.

All nine measured cache-on repetitions again preserve exact prompt/generated
tokens, all ordered router selections, cache-to-router correspondence, event
ordering, independent LRU state, generations, mappings, process cleanup, and
determinism.

- exact consolidated arena: 963,483,264 bytes;
- measured events: 8,235;
- demands: 65,880;
- hits: 46,848;
- misses: 19,032;
- aggregate hit rate: 71.11%;
- bytes transferred: 63,669,881,184;
- aggregate host-wall blocking time: 16,534.323 ms;
- mean blocking per run: 1,837.147 ms;
- blocking per generated token: 35.10 ms, including prefill transfers;
- cache-on sampled system GPU peak: 4,056 MiB maximum;
- measured system-wide reserve at that peak: 12,255 MiB of 16,311 MiB.

The lower cache-on system peak is expected because nine full expert tensors are
moved out of the CUDA model allocation and replaced by a smaller 32-slot arena.
It is a system-wide sample, not a process-specific allocator measurement.

Per-layer results:

| Layer | Hit rate | Misses | Bytes transferred | Blocking ms |
|---:|---:|---:|---:|---:|
| 21 | 74.63% | 1,857 | 6,212,430,084 | 1,674.839 |
| 22 | 76.02% | 1,755 | 5,871,198,060 | 1,657.544 |
| 23 | 76.52% | 1,719 | 5,750,763,228 | 1,614.369 |
| 24 | 72.21% | 2,034 | 6,804,568,008 | 1,772.197 |
| 25 | 72.91% | 1,983 | 6,633,951,996 | 1,747.295 |
| 26 | 69.10% | 2,262 | 7,567,321,944 | 1,916.640 |
| 27 | 67.25% | 2,397 | 8,018,952,564 | 2,000.214 |
| 28 | 62.62% | 2,736 | 9,153,047,232 | 2,185.034 |
| 29 | 68.73% | 2,289 | 7,657,648,068 | 1,966.191 |

Focused means:

| Mode | Prompt TPS | Decode TPS | End-to-end ms | Peak GPU MiB |
|---|---:|---:|---:|---:|
| current cache off | 22.619 | 26.853 | 4,184.523 | 6,743 max |
| all eligible reactive | 13.212 | 19.646 | 6,466.734 | 4,056 max |

Relative to current cache-off, prompt TPS is -41.59%, decode TPS is -26.84%,
and end-to-end time is +54.54%. Most of the end-to-end increase is explained by
the measured blocking copies and synchronization rather than cache metadata.

For context, the committed diagnostic means were 98.53 decode TPS for strongest
stock no-OOM, 26.50 for matched stock `-ngl 10`, and 25.25 for canonical
observer/cache-off. All-eligible reactive reaches 19.65 decode TPS: 80.06%
behind strongest stock, 25.86% behind matched stock, and 22.19% behind the
committed observer result.

## Stop decision

The all-eligible configuration is exact and memory-safe, but it is not
performance-credible. Broader coverage multiplies blocking misses and
synchronization cost faster than it creates useful runtime savings.

Therefore:

- do not test `-ngl 15` or `-ngl 20`;
- do not test 64 slots;
- do not integrate B2 prediction into this operating point;
- preserve the two-layer and all-eligible exactness evidence;
- keep the one-layer C5 result as the strongest current ExpertFlow runtime
  result;
- keep live caching disabled by default.

Host-wall blocking durations remain copy-plus-synchronization measurements.
They are not CUDA-event latency or copy/compute overlap.

## Evidence

- two-layer root:
  `C:\models\expertflow\runs\cuda-eligible-cache\two-layer-focused`
- all-eligible root:
  `C:\models\expertflow\runs\cuda-eligible-cache\all-eligible-ngl10-focused`
- auto discovery:
  `C:\models\expertflow\runs\cuda-eligible-cache\auto-ngl10-smoke`
- explicit rejection:
  `C:\models\expertflow\runs\cuda-eligible-cache\reject-layer0`

SHA-256:

- two-layer command ledger:
  `a6fe467240d50abcc1c209617e948fc0afdaccf49b2ca6999a1bbc6e5e986c82`
- all-eligible command ledger:
  `1e954d7bd01a26ba8a0adf38496be18d4699b1f17724b3d2ad1cf3fc539a960d`
- auto-discovery log:
  `4c89c0ad153196c7002a3b54666910fcb18d01b319f85aa3761a03689401f82a`
- probe:
  `06e72c805a1027c7e44bbe6b90fc24f8cd111bdd9808b6d7a14538dec9e503d5`
- `ggml-base.dll`:
  `e515ad81351f4beb6215cafc16371d88be1df293af4aa5021d4b6bb9ed8bff04`
