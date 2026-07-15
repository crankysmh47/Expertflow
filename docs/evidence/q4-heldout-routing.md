# Held-out Q4 routing and recommendation evidence

> **Quarantined historical evidence:** These real-model traces use the callback now labeled `trace_v1_perturbing`. They remain available for provenance but are excluded from final locality and cache-policy claims until recollected with an exactly transparent observer. See `configs/trace-evidence-status.json`.

Date: 2026-07-15 PKT

- **Training workload:** original five parity-safe Vulkan prompt-prefill traces
- **Evaluation workload:** five new prompts collected after the training workload and policy curve were frozen
- **Backend:** official llama.cpp b10002 Vulkan on NVIDIA GeForce RTX 5060 Ti
- **Parity:** PASS for all five held-out tracing-disabled/tracing-enabled pairs
- **Policy evidence:** held out
- **Runtime gate:** CONDITIONAL; live cache disabled

## Held-out prompt shapes

| Slug | Prompt |
| --- | --- |
| `json-deployment-risk` | Return a JSON object with risk, evidence, and action for a latency-only deployment incident. |
| `cache-arithmetic` | Calculate a layered cache allocation from slots and per-expert MiB, then state an assumption. |
| `prefetch-comparison` | Compare optimistic prefetch and conservative admission in a constrained table. |
| `urdu-transfer` | Translate a cache-miss statement into Urdu and explain the transfer term in English. |
| `pcie-diagnosis` | Diagnose stable routing, low compute utilization, and per-token PCIe spikes. |

Every pair used the same verified Q4 model, unchanged probe, 10 offloaded layers, 12 threads, greedy sampling, and eight generated tokens. All ten inference processes exited 0 in 66.7 seconds. Prompt and generated token IDs matched exactly in all five pairs.

The held-out traces contain 5,640 full events and 45,120 expert demands. Filtering to fixed-prompt prefill and target layers 0–20 leaves 3,213 evaluation events and 25,704 demands. The frozen training set contributes 2,688 events and 21,504 demands under the same filter.

## Frozen-policy curve

Static residents are selected only from the original training traces. They are then evaluated without refitting on the five held-out traces. LRU starts empty and adapts online only within the held-out sequence.

| Slots/layer | Projected cache | Frozen static hit rate | Held-out LRU hit rate | Static serialized H2D/token | LRU serialized H2D/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 536.09 MiB | 30.89% | 36.28% | 27.29 ms | 25.16 ms |
| 16 | 1,072.19 MiB | 47.28% | 52.37% | 20.82 ms | 18.81 ms |
| 32 | 2,144.38 MiB | 66.80% | 69.90% | 13.11 ms | 11.88 ms |
| 64 | 4,288.76 MiB | 86.61% | 85.82% | 5.29 ms | 5.60 ms |
| 75 | 5,025.89 MiB | 91.11% | 89.07% | 3.51 ms | 4.32 ms |
| 96 | 6,433.14 MiB | 96.45% | 91.55% | 1.40 ms | 3.34 ms |
| 108 | 7,237.28 MiB | 97.26% | 91.76% | 1.08 ms | 3.25 ms |

The static-96 estimate falls from 99.57% in-sample to 96.45% held out. Static placement does not overtake held-out LRU until 64 slots/layer. Capacity 108 still exceeds the measured 7,234 MiB configurable headroom before cache-specific workspace.

The serialized H2D columns remain transfer-only estimates. They use held-out miss counts and the measured 0.235042 ms pinned transfer mean per expert, but do not establish whether transfers meet layer deadlines.

This table evaluates held-out prompt prefill. The later [decode deadline checkpoint](q4-deadline-oracle.md) trains on prefill and evaluates untouched decode events. Static-96 reaches 93.28% in that cross-phase test and is the current conservative recommendation input.

## Artifacts

- Held-out curve: `C:\models\expertflow\runs\heldout-q4-vulkan\heldout-curve-cpu21.json`, 8,862 bytes, SHA-256 `3f93aa31897427e50bf0b3c08006176d5041072b62d22a26ba3868f8a48384bf`
- Recommendation: `C:\models\expertflow\runs\q4-probe\recommendation-heldout.json`, 1,813 bytes, SHA-256 `947f86ee5d0900b1c2493de1b4a53f080f62e278585bafce5b1d4f776bd9155a`
- Standalone replay: `C:\models\expertflow\runs\q4-probe\report-heldout.html`, 51,181 bytes, SHA-256 `028e0fcb4526d54f57190549c3e391af96d305b556d286c5c77e31584a871840`

The report replays the held-out events with the frozen training hotset. It reconciles exactly to 24,791 ready and 913 blocking selections, lists training and evaluation sources separately, contains no scripts, and loads no remote assets.

## Decision

Static-96 remains the highest tested point inside the measured envelope: 6,433.14 MiB projected cache and 800.86 MiB configurable headroom left beyond the separate 1,024 MiB safety reserve. Its evidence label is now held out rather than in-sample. The current decode-focused recommendation uses the cross-phase 93.28% result rather than the stronger prefill-only result.

The verdict remains `CONDITIONAL` and `live_cache_enabled=false`. Held-out policy evidence is no longer a blocker. The remaining blockers are:

- per-layer CUDA compute deadlines are not measured;
- no exact same-runtime reactive-versus-cached end-to-end comparison exists.
