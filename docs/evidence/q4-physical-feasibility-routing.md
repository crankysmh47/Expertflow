# Q4 physical-feasibility routing evidence

> **Quarantined historical evidence:** The 40-conversation corpus was collected through the callback now labeled `trace_v1_perturbing`. Preserve it, but exclude it and every derived policy result from final claims. Do not collect additional shards until tracing parity is restored. See `configs/trace-evidence-status.json`.

Date: 2026-07-15 PKT

This checkpoint expands the locality probe from ten historical prompts to a frozen 40-conversation corpus. The corpus is checked in at `configs/q4-physical-feasibility-corpus.json`; raw traces and process logs remain under `C:\models\expertflow\runs\physical-feasibility-q4-vulkan`.

## Collection contract and result

- Model: verified Gemma 4 26B A4B Q4_0, SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- Runtime: pinned llama.cpp b10002 Vulkan probe, revision `a7312ae94f801fc9c6786dc56e38df57b964f697`
- Split: 32 train, 4 validation, 4 test, frozen by complete conversation before inference
- Domains: general chat 10, code 8, math/reasoning 6, translation 4, multilingual 4, long-context 4, structured output 2, topic shift 2
- Decode: greedy, 64 requested output tokens, 10 GPU layers, 12 CPU threads
- Pairing: a separate tracing-disabled and tracing-enabled process for every conversation
- Runtime: 827.886 seconds across the first 80 native processes
- Latest-attempt trace volume: 152,760 events and 1,222,080 expert demands

The first pass produced 39 exact-parity pairs and one deterministic failure. `train-translation-02` returned exit 0 in both modes and matched all prompt tokens, but generated token 0 was `100` without tracing and `20663` with tracing. A separate retry produced the same token files and mismatch. The original and retry remain in separate immutable attempt directories. This failed training shard is not used for policy fitting and is listed in every downstream artifact; all eight validation/test pairs passed.

Collection manifest: `C:\models\expertflow\runs\physical-feasibility-q4-vulkan\collection-manifest.json`, 243,939 bytes, SHA-256 `47e2154870c50b2aed9aef148a7b9a6496173d2a1516529c744fa7f5a1981093`.

## Held-out static-96 results

Static residents are selected by per-layer frequency from the 31 parity-safe training traces only. LRU starts empty for each held-out conversation. Both policies use 96 slots per target layer over layers 0–20. Prefill and decode are reported separately; the decode result fits static residents on training prefill and evaluates held-out decode.

| Held-out conversation | Domain | Prefill static / LRU | Decode static / LRU |
| --- | --- | ---: | ---: |
| `validation-general-10` | General chat | 97.34% / 84.55% | 94.12% / 86.97% |
| `validation-code-08` | Code | 95.64% / 83.80% | 85.81% / 90.05% |
| `validation-math-06` | Math/reasoning | 94.01% / 80.95% | 87.58% / 83.50% |
| `validation-translation-04` | Translation | 95.71% / 78.73% | 88.69% / 85.66% |
| `test-multilingual-04` | Multilingual | 95.85% / 83.33% | 87.47% / 87.03% |
| `test-long-04` | Long context | 96.92% / 91.94% | 88.51% / 85.47% |
| `test-structured-02` | Structured output | 95.72% / 86.52% | 85.80% / 86.39% |
| `test-shift-02` | Topic shift | 95.14% / 91.78% | 82.60% / 85.64% |
| **Weighted aggregate** | **Eight conversations** | **95.85% / 87.48%** | **87.57% / 86.34%** |

The prefill evaluation contains 13,503 layer events and 108,024 expert demands. The decode evaluation contains 10,584 layer events, 84,672 demands, and 504 complete decode forwards. The probe requests 64 generated tokens but does not decode the final sampled token again, so each held-out conversation contributes 63 decode forwards.

Decode is the conservative policy result. Static-96 has 10,523 misses versus 11,567 for reset-per-conversation LRU, a 9.03% cold-byte reduction. Static loses to LRU on code, structured output, and topic shift. This is materially weaker than the earlier five-prompt result and does not meet the product-spec gate of at least 20% lower cold bytes or 25% lower estimated PCIe stall than per-layer LRU.

Structured artifacts using the pooled two-slice p95 transfer value:

- Prefill: `heldout-breakdown-prefill-static96-p95.json`, 19,586 bytes, SHA-256 `24eb45f64a9e4165b1f5ba0e94d91ce7aa1fbb9fde6c7be337aca99e13872b6b`
- Decode: `heldout-breakdown-decode-static96-p95.json`, 19,618 bytes, SHA-256 `b887438a9a18e5f730d8d2a5328f9b7dec69417802c7a5ea0ee2489da00be69d`

These are estimated policy outcomes over measured routing. They are not live cache hits or speedup measurements. `live_cache_enabled=false` remains unchanged.
