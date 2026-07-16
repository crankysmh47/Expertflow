# Nine-layer cache performance isolation

**Verdict:** blocking transfers are the dominant measured cost, detailed event
instrumentation is not, but the exact warm-static proof is infeasible under the
frozen 32-slot/direct-kernel constraints. Stop before predictive asynchronous
integration.

This experiment preserves `-ngl 10`, eligible layers `21..29`, 32 slots per
layer, the fixed general/code/translation prompts, one warmup plus three
measured repetitions, 64 requested generated tokens, exact routing traces, and
100 ms system GPU sampling. No higher offload, 64-slot cache, prediction,
asynchronous copy, graph relocation, kernel change, or speedup claim was added.

## Committed starting points

- llama.cpp CUDA-resident eligibility: `f9231b02`
- ExpertFlow evidence/runtime harness: `2e8bb2b`

Neither commit was merged into a protected or canonical branch and neither was
pushed.

## D0, D2, and D3

D0 is cache-disabled. D2 is the committed detailed-event reactive LRU. D3 uses
the identical reactive cache and internal exactness checks, but skips
per-event-record construction and writes only one deferred aggregate record per
layer.

All nine D3 measured repetitions match D0 and D2 exactly for prompt tokens,
generated tokens, and every ordered router selection. D2 and D3 also match
exactly on 8,235 events, 65,880 demands, 46,848 hits, 19,032 misses,
63,669,881,184 transferred bytes, and a 71.11% hit rate.

| Mode | Prompt TPS | Decode TPS | End-to-end ms | Peak GPU MiB |
|---|---:|---:|---:|---:|
| D0 cache disabled | 22.581 | 26.893 | 4,185.328 | 6,716 mean |
| D2 detailed reactive | 13.212 | 19.646 | 6,466.734 | 4,051 mean |
| D3 aggregate reactive | 13.336 | 19.708 | 6,424.656 | 3,959 mean |

D2 relative to D0:

- prompt TPS: -41.49%;
- decode TPS: -26.95%;
- end-to-end time: +54.51%;
- measured blocking: 1,837.147 ms/run.

D3 relative to D0:

- prompt TPS: -40.94%;
- decode TPS: -26.72%;
- end-to-end time: +53.50%;
- measured blocking: 1,822.663 ms/run.

D3 relative to D2:

- prompt TPS: +0.94%;
- decode TPS: +0.32%;
- end-to-end time: -0.65%.

Detailed event recording is therefore not a severe instrumentation penalty.
For D3, measured blocking accounts for 81.39% of the 2,239.327 ms/run
end-to-end increase over D0. Removing that measured blocking arithmetically
leaves 416.664 ms/run, or 9.96% over D0. This subtraction is diagnostic only;
it is not proof that asynchronous copy would recover the time.

## D1 warm-static feasibility

Training-only top-32 sets were derived from the frozen seven training
conversations in `trace_v2_canonical_segmented_pilot`. On the fixed benchmark
prompts they cover:

| Prompt | Training-static hit rate | Static misses/run | Unique experts/layer |
|---|---:|---:|---:|
| general | 45.62% | 4,111 | 92-110 |
| code | 43.85% | 4,690 | 91-107 |
| translation | 52.03% | 2,901 | 85-100 |

No 32-expert fixed set can cover any full measured prompt: each eligible layer
touches at least 85 and as many as 110 unique experts.

The unchanged CUDA operation consumes one direct 32-slice tensor. Exact fallback
without modifying the fixed cache would require selecting between that arena
and the original 128-expert tensor per event, or adding a separate fallback
operand. Both require a graph or kernel execution-path change prohibited by the
experiment. Allowing replacement would turn D1 back into D2. Silently executing
stale or missing experts would violate exactness.

D1 was therefore classified as physically infeasible under the requested
simultaneous constraints, not run with weakened correctness.

## Decision

The evidence supports these conclusions:

1. synchronous copies and synchronization dominate the nine-layer regression;
2. deferred detailed logging is not the main problem;
3. the current evidence does not prove that execution through a warm 32-slot
   arena is close enough to baseline;
4. at least about 10% end-to-end overhead remains after diagnostic subtraction
   of measured blocking;
5. predictive asynchronous integration is not authorized by the stated
   decision rule because D1 could not establish the required warm-static
   execution result.

Stop before predictive integration. A future isolation may proceed only with a
separately approved exact fallback design or a controlled all-hit workload
whose relationship to the fixed benchmark is explicitly limited. Remaining
categories are logical-to-physical ID rewriting, 32-expert CUDA kernel
behavior, mandatory synchronization, tensor-layout/memory-access effects, and
aggregate cache bookkeeping.

### Subsequent authorization

After reviewing this evidence, the user accepted it as sufficient to proceed
with bounded predictive asynchronous integration. The authorization explicitly
waives the earlier requirement for a 32-slot no-miss warm-static control while
retaining exactness, memory stability, genuine CUDA asynchronous-transfer
measurement, and residual-overhead tracking as mandatory gates.

`live_cache_enabled=false` remains the default. No runtime speedup is claimed.
Host-wall blocking is not CUDA-event latency or copy/compute overlap.

## Evidence roots

- D0/D3:
  `C:\models\expertflow\runs\cuda-eligible-cache\isolation-d0-d3`
- D2:
  `C:\models\expertflow\runs\cuda-eligible-cache\all-eligible-ngl10-focused`
- manifest:
  `C:\models\expertflow\runs\cuda-eligible-cache\isolation-d0-d3-manifest.json`
- machine-readable summary:
  `docs/evidence/live-cache/performance-isolation-d0-d3.json`

The first manifest-generation command failed because Windows PowerShell did not
support `ConvertFrom-Json -AsHashtable`. The corrected command initially wrote
a UTF-8 BOM, which Python rejected. Rewriting the generated manifest as UTF-8
without BOM resolved the command-layer issue; no model process ran during
either failed attempt.

The benchmark runners now also remove inherited
`EXPERTFLOW_LIVE_CACHE_LOG_DETAIL` before applying each mode's explicit
environment. This prevents a parent shell from silently changing a detailed
D2 run into aggregate-only D3.
