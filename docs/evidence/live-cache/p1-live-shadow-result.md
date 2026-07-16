# P1 live-shadow predictor result

P1 passes the bounded live-shadow gate for the frozen B2 transition predictor
from MoE layer 23 to layer 24. It makes no cache, residency, eviction, transfer,
or expert-selection decision. `live_cache_enabled=false` remains the default.

## Frozen predictor

- P0 commit: `6bc8eb68f835a3588e97704595139cf5c249faf2`
- family: B2 transition tables
- scoring: source-normalized
- tables: phase-separated
- candidate width: 12
- admission: observed support
- tie break: ascending expert ID
- runtime artifact: 267,556 bytes
- artifact SHA-256:
  `54f898ec25fc4b783953f8c98ffb122073e91741b31b94aef2e285d26063409b`
- payload SHA-256:
  `8837f31178e1b049f23e6ff2ad1654908055b1d8ae79c19994a33df0a6424f40`

The artifact stores float64 phase tables, support masks, fixed dimensions,
model/runtime/corpus/configuration identifiers, and a payload checksum. The
native loader rejects incompatible identities, schema or dimension changes,
truncation, corruption, non-finite values, invalid source IDs, duplicate source
IDs, and unsupported phase values.

The runtime identifier in the artifact is the canonical observer compatibility
identifier. The newly built P1 binary is pinned separately below, avoiding a
circular self-hash requirement.

## Runtime path

The phase is set explicitly immediately before one `llama_decode` call. The
scheduler consumes a monotonically increasing generation once and the public
decode wrapper clears it on every return path. Missing, invalid, contradictory,
or stale phase state fails closed.

P1 reuses the canonical full-trace callback's already-materialized layer-23 and
layer-24 ID arrays. It adds one narrow call from that existing callback into the
runtime scorer. It does not register another callback, request another tensor,
call a tensor getter, add synchronization, or alter graph construction or
scheduling. Shadow-disabled and shadow-enabled measurements therefore use the
same observer path.

Prediction uses stack-only fixed arrays and `std::sort`; records use one
preallocated 8,192-entry arena. File output and formatting occur only during
scheduler teardown. Overflow, incomplete transitions, wrong ordering, and
generation mismatches fail explicitly.

## Validation

The focused suite used the frozen general, code, and French-translation prompts
with one warmup and three measured repetitions per mode. The seven-task smoke
suite then covered code, arithmetic, reasoning, structured JSON, translation,
and concise general output.

| Gate | Result |
| --- | --- |
| Focused warmup pairs | 3 passed |
| Focused measured pairs | 9 passed |
| Seven-task smoke pairs | 7 passed |
| Live transitions | 1,101 |
| Prompt/generated-token parity | exact |
| Ordered router parity | exact |
| Offline/live candidate IDs | exact |
| Offline/live float64 scores | exact |
| Candidate-support failures | 0 |
| Pending/overflowed records | 0 |
| Repeated token determinism | exact |
| Paired router event counts | exact |
| Predictor-driven weight transfers | 0 |
| Live-cache mutations | 0 |

Measured live results:

| Phase | Transitions | Recall@8 | Recall@12 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill | 875 | 53.77% | 66.90% | 7.6 us | 8.5 us |
| Decode | 226 | 50.77% | 63.00% | 7.5 us | 8.375 us |

The live prompt set differs from the sealed expanded test, so these recall
values do not replace the frozen P0 sealed-test results.

Across the nine focused measured pairs, shadow mode versus the matched
observer-enabled binary measured:

- prompt TPS: -0.286%
- decode TPS: -0.187%
- end-to-end time: +0.272%
- time to first token: +0.293%

These differences are within run-to-run variance and do not support a speedup
claim. Global peak GPU use was identical in the full aggregate: 6,680 MiB.
Settled GPU use returned to its pre-run value after every process. Mean peak
working set increased by about 155 KiB, consistent with the fixed record and
artifact storage; no persistent host or GPU growth was observed.

## Pinned artifacts

| Artifact | SHA-256 |
| --- | --- |
| `expertflow-router-probe.exe` | `c2cc7d91ad3c544daa232de81a4119a27cc4a8aaad7c702971b324fc32227fdb` |
| `llama.dll` | `cc01af39da463c796f513327fe991f6077b8c372bc78758301c22f11f495e581` |
| `ggml-base.dll` | `70abb80b23e5e2809e0e5d78c4f5357db4ec362ecbdf80be1c53f30e942a52a6` |
| `ggml-cpu.dll` | `99382381341c874d51b277018a68ca76985e882962f37cda810cd3dc0e45a792` |
| `ggml-cuda.dll` | `6533db2aa0bcc2503dd5758591aca37ace6c7cfd2a3c3d47109482b9cdab3fa4` |
| aggregate JSON | `914cd215ce4946b54fe6b9b7de97ea4430faacea86e3ea446cef9ecd5948ba9d` |

Toolchain: MSVC 19.39.33523, CUDA 12.8, CMake 4.3.1, Ninja 1.13.2,
NVIDIA driver 591.86, and RTX 5060 Ti with 16,311 MiB.

## Regression gates

- 139 ExpertFlow tests passed.
- Native predictor tests passed with assertions active.
- Native exact-cache tests passed.
- Judge replay reproduced 8 events, 64 demands, 26 static-hotset hits, and
  19 LRU hits.
- No ExpertFlow environment variables or model processes remained.
- Both worktrees passed `git diff --check`.

P1 supports issuing a P2 design. It does not authorize P2 implementation.
