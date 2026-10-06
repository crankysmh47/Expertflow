# Temporal layer-24 next-token predictor

## Verdict

T0 passes as bounded offline feasibility evidence and supports proceeding to
T1 live shadow. It does not yet authorize predictive transfers or a speedup
claim.

The frozen validation split selected the combined temporal scorer with weights
`0.50 transition / 0.40 current-set retention / 0.10 causal session frequency`
and candidate width 16. The sealed temporal test split was opened exactly once.
The selected configuration remained positive across all six domains and
improved the all-ready simulated cache over reactive LRU, but it also produced
substantially more wasted than useful speculative bytes. Therefore T1 should
measure candidate overlap and token-to-token lead time without moving weights;
any later T2 experiment must retain the approved one-transfer-per-token limit.

## Frozen contract

- Manifest:
  `C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json`
- Manifest SHA-256:
  `b59ab020843a98121fca1f60227d3bf7272fe2a0e8f4b10c53add7182f7437fa`
- Conversations: 60 train / 12 validation / 12 sealed test
- Domain balance: 10/2/2 in each of six domains
- Layer: 24 only
- Phase: decode only
- Join: same conversation/request/turn, exact next forward and token index,
  increasing causal hook order
- Decode layer-24 events: 2,594
- Consecutive temporal samples: 2,510 total; 1,790 train, 360 validation, 360
  test
- Observed join gaps: zero

The prior adjacent-layer test result was not used for temporal selection. The
temporal selection lock records
`adjacent_layer_test_used_for_selection=false`.

## Fixed validation search

The search was declared before test access:

- T0.0 copy
- T0.1 causal session frequency
- T0.2 source-normalized temporal transitions
- T0.3 combined score with four fixed weight triples
- widths 8, 12, and 16

The deterministic objective maximized simulated ready improvement after
eviction regret, then minimized waste, maximized recall, minimized latency,
preferred narrower width, and preferred simpler policies.

Selected validation result:

| Metric | Result |
|---|---:|
| Samples | 360 |
| Recall@8 | 44.31% |
| Recall@12 | 58.23% |
| Recall@16 | 66.15% |
| Exact top-8 set match | 0.00% |
| CPU prediction p50 / p95 | 54.2 / 62.0 us |
| Reactive misses | 675 |
| Simulated misses covered | 98 |
| Eviction regret | 21 |
| Net gain after regret | 77 |
| Useful / wasted insertions | 73 / 349 |
| Useful / wasted projected bytes | 244.22 MB / 1,167.55 MB |

The lock was written with `test_opened=false`, all 12 original test identities,
and immutable payload SHA-256
`dbc179804ac3f33c6ba057f32029bae11847dba42a8d71310134b2df7868c32c`.

## Single sealed-test result

| Metric | Result |
|---|---:|
| Samples | 360 |
| Recall@8 | 47.01% |
| Recall@12 | 60.52% |
| Recall@16 | 67.22% |
| Exact top-8 set match | 0.00% |
| CPU prediction p50 / p95 | 59.0 / 80.4 us |
| Reactive hits / misses | 2,255 / 625 |
| Simulated ready improvement | 110 |
| Simulated miss reduction | 17.60% |
| Eviction regret | 17 |
| Net gain after regret | 93 |
| Useful / wasted insertions | 70 / 355 |
| Useful / wasted projected bytes | 234.18 MB / 1,187.62 MB |

Recall by domain:

| Domain | Samples | Recall@8 | Recall@12 | Recall@16 |
|---|---:|---:|---:|---:|
| Code | 60 | 46.88% | 62.50% | 71.67% |
| General instruction | 60 | 39.79% | 55.83% | 62.92% |
| Math/reasoning | 60 | 51.67% | 66.67% | 73.13% |
| Structured output | 60 | 47.50% | 60.21% | 65.42% |
| Topic shift | 60 | 43.33% | 55.83% | 63.13% |
| Translation/multilingual | 60 | 52.92% | 62.08% | 67.08% |

These cache figures are simulated under an all-ready timing model. They are not
CUDA-event latency, transfer overlap, live cache behavior, or throughput.
Width 16 was selected for aggregate simulated utility; it is not permission to
transfer 16 experts. The next live transfer stage remains capped at one
highest-ranked nonresident candidate per token.

## Reproducibility and artifacts

```powershell
$env:PYTHONPATH='src;.'
uv run python scripts\run_temporal_layer24_predictor.py fit `
  --manifest C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json `
  --output C:\models\expertflow\runs\temporal-layer24-predictor

uv run python scripts\run_temporal_layer24_predictor.py test `
  --manifest C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json `
  --output C:\models\expertflow\runs\temporal-layer24-predictor
```

The first fit failed before lock creation because a zero-valued simulator
counter was absent. The exact failure is retained in `ledger.jsonl`; a
regression test now requires complete zero-valued metric schemas. The corrected
fit took 1.51 seconds and the sole test evaluation took 0.38 seconds. A second
test command failed closed, and the test-metrics SHA-256 remained unchanged.

Key artifact hashes:

- selected predictor:
  `459e78db7635e128a38c195c1930672f017604c916959ba40adbb9dc1985e5ba`
- validation metrics:
  `ae42df7b2858350443ac766fda9f82cf8bbcadada39dc9bcb49d471089d992f6`
- test metrics:
  `5a795aefd90ef4c150fe65924e461f652f94cfc303317a11433376f0c5d97178`
- final selection lock:
  `88485f13132128ae17a9227537ad5d5a6f774b70418b01e2102969c89678ead7`
- judge replay:
  `40cd90a1bc45f3e65c4293eb41c0646d248bc86dbf7e4223d8efb19804168139`

Final verification passed 160 ExpertFlow tests. Judge replay reproduced eight
events, 64 demands, 26 static-hotset hits, and 19 LRU hits. `git diff --check`
passed, the temporal llama.cpp branch remained clean, and no llama/router
process remained after validation.

## Next gate

Proceed to T1 live shadow only:

1. consume authoritative layer-24 decode IDs at token `t`;
2. update causal session state;
3. predict token `t+1` candidates without cache mutation or transfer;
4. compare them with the next authoritative layer-24 event;
5. measure prediction latency, overlap, reset behavior, and the real
   token-to-token lead-time distribution;
6. require exact shadow-off/on tokens and router parity.

Do not begin T2 unless T1 proves deterministic offline/live equivalence and a
practical opportunity for at least one ready-useful next-token transfer.
