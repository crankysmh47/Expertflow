# Canonical Observer v1 sanity smoke

## Decision

**ACCEPT** the graph-segmenting Observer v1 as the canonical ExpertFlow hackathon runtime. Cache remains disabled. This decision does not claim equivalence to untouched llama.cpp or a runtime speedup.

The hackathon runtime uses graph-segmenting routing telemetry. This changes the numerical execution path and can produce alternative near-boundary expert selections. ExpertFlow uses the same instrumented runtime for data collection, cache-disabled baselines, cache-enabled runs, and evaluation.

## Results

| Task | Mode N | Mode O | O tokens | O routing events |
|---|---:|---:|---:|---:|
| Python palindrome | pass | pass | 89 | 4,350 |
| Python merge intervals | pass | pass | 176 | 6,840 |
| Arithmetic components | pass | pass | 9 | 1,980 |
| Request-rate reasoning | fail (`150`) | pass (`120`) | 9 | 2,310 |
| Strict JSON | pass | pass | 41 | 3,630 |
| French translation | pass | pass | 29 | 2,520 |
| Reproducibility bullets | pass | pass | 28 | 2,250 |

Mode N passed 6/7, Mode O passed 7/7, and Mode O retained all six successful Mode N outcomes. Both generated code blocks passed the fixed local test cases. JSON parsed to the exact required object. Translation preserved server restart, ordering after the update, contrast, and absence of user-data loss. The three bullets were relevant and at most six words each.

The merge output budget was increased after the first Mode N output was truncated. The JSON prompt received the small explicit suffix `Do not use Markdown code fences.` after the first Mode N response used a fence. No result was reinterpreted: Mode N's request-rate answer remains recorded as an objective failure.

## Determinism and stability

Two fresh Mode O executions with the final binary produced identical generated-token files and identical 1,980 routing records when the intentionally variable `observed_at_ns` field was excluded. All 14 suite processes exited successfully. Per-run settled GPU snapshots returned to their pre-run state; no persistent probe process or allocation growth was observed. Process-specific VRAM was unavailable under WDDM and is recorded as zero/unsupported rather than treated as a measurement.

Wall-clock durations are diagnostic metadata only. They support no performance claim.

## Identity

- ExpertFlow base: `8c3cef0b5cd8aa5fe86ef66282327cd1534aca0c`
- llama.cpp: `a7312ae94f801fc9c6786dc56e38df57b964f697`
- Canonical probe SHA-256: `7ea12e0c44258bb2d75f99b7e180e7ef7cb0c7d3e285ca231fe26ad8c8c4932c`
- Model SHA-256: `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- Machine summary: `C:\models\expertflow\runs\canonical-observer-smoke\suite\summary.json`
- Summary SHA-256: `9fd377e1dae4ba3c6b439b0eadab1b2dc250903c2a16382a50da236a1f23c859`
- Append-only command ledger SHA-256: `ca08df650994e393cbe1310b2f7b8003b6bc923a4a1f297223da025bbaac8b83`
- Observer v2 recovery stash: `bb8ee4522f28dafd1b819747d58f7198d0a3a038`

Old `trace_v1_perturbing` traces remain quarantined. New eligible traces use `trace_v2_canonical_segmented` and must be compared only within this observer-enabled distribution.
