# Canonical segmented-trace pilot locality

The `trace_v2_canonical_segmented` pilot collected 14/14 successful conversations and 50,310 routing events using the pinned observer-enabled/cache-disabled runtime. Seven training conversations and seven untouched validation/test conversations cover general chat, code, math/reasoning, translation, multilingual, structured output, and topic shift. Raw traces remain preserved beside canonical traces whose only transformation is replacing placeholder request/conversation IDs.

Decode adjacent-token selected-expert reuse is **39.44%** across 49,920 comparable expert demands.

| Slots per layer | Training-only static | Reset LRU | Session frequency | Hindsight session-static bound |
|---:|---:|---:|---:|---:|
| 8 | 19.28% | 38.15% | 39.16% | 47.02% |
| 16 | 31.66% | 55.94% | 56.10% | 68.32% |
| 32 | 49.34% | 72.30% | 71.83% | 88.90% |
| 64 | 75.88% | 78.40% | 78.35% | 99.58% |
| 96 | 92.80% | 78.49% | 78.49% | 100.00% |

Static residents were selected exclusively from training traces. Every reported policy score uses only validation/test conversations, and LRU/session state resets per conversation. Session frequency is causal. The hindsight column is explicitly non-causal and is only an upper bound for a fixed per-conversation set.

This small pilot confirms useful locality and supports a bounded exact live-cache experiment. It is not a final generalization result. The new 92.80% pilot result does not restore or validate the withdrawn 93.28% historical claim. Policy results are estimates over measured routing; `live_cache_enabled=false`, and no runtime speedup is claimed.

Artifacts:

- Collection manifest: `C:\models\expertflow\runs\trace-v2-canonical-segmented-pilot\collection-manifest.json`, SHA-256 `ec4f43fe7cb5a6ed8ebfcab147cd9e9925762a17d0b03286406e7e67fb5268db`
- Held-out curve: `C:\models\expertflow\runs\trace-v2-canonical-segmented-pilot\heldout-curve-decode.json`, SHA-256 `86c9612df8f822d160ffe727c1d7c67bc2b0cd4cb49b30790cf53e2c624c6b3a`
- Policy summary: `C:\models\expertflow\runs\trace-v2-canonical-segmented-pilot\policy-summary-decode.json`, SHA-256 `1cc01791ad82693e4605dee238eb68c0614739c7c28c0fb39bcc9f09a3d1f8a3`
