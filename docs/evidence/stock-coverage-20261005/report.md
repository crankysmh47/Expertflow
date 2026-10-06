# Wider stock utility results

Collected 2026-10-06 on `ef-v2` from `78ad5f0063f2bc372a528e37ec1f45d8be1ceb11`. Terminal verdict: **COMPLETE-STOCK-COVERAGE**, exit0, **344 retained attempts/native starts**. All four cases are **NO-UTILITY-GAIN**. No case met the registered practical gain threshold; no conditional product or consumer ran. The 84 unused conditional calls cannot fund retries or further candidates.

The separate collector passed one implementation review, all six material fixes, **891 tests with seven expected historical-source skips**, six applicable pinned native source checks and seven installed-wheel checks before collection. It froze 63 exact raw source/input files, all four live identities, roots, eligibility/defaults proofs and schedules. The immutable [registration](../../superpowers/specs/2026-10-05-stock-coverage.md) and original Q6 studies remain unchanged.

## Four separate comparisons

Defaults are 8 threads/CUDA graphs on, with the registered placement, CPU-MoE, workload and F16 settings fixed. Every automatic/manual search used 18 native evaluations across the same six thread/graph candidates. The fixed utility gate requires at least 5% paired geometric decode-TPS gain and a positive CI95 lower bound, plus manual equivalence and the other numerical/resource gates.

| Registered case | Automatic / manual threads; graphs on | Gain over defaults; CI95 | Automatic vs manual CI90 | Verdict |
| --- | --- | --- | --- | --- |
| Gemma Q4 prose | 12 / 12 | +1.03%; [+0.88%, +1.19%] | [−1.01%, +0.11%] | NO-UTILITY-GAIN |
| Gemma Q4 code | 12 / 12 | +0.78%; [+0.63%, +0.93%] | [−0.20%, +0.08%] | NO-UTILITY-GAIN |
| Granite Q6 prose | 8 / 16 | −0.02%; [−0.15%, +0.09%] | [−0.12%, +0.04%] | NO-UTILITY-GAIN; selected default |
| Granite Q6 code | 8 / 16 | −0.23%; [−0.38%, −0.08%] | [−0.20%, +0.04%] | NO-UTILITY-GAIN; selected default |

All manual-equivalence intervals lie strictly inside ±2%, and native evaluation cost is equal at 18/18. Q4's positive changes fall below the frozen 5% gate; they are not qualified practical utility. Granite selected the default, and its confirmation blocks measure independent executions of that same configuration. No negative result was discarded or used to retune the registered space. Each interval belongs to its case, with no pooling or family-wide guarantee.

## Costs and provenance

| Case | Calls | Input load / collection / reconstruction seconds | Fixed waits seconds | Native phases seconds | Peak owned bytes / minimum device-free bytes |
| --- | --- | --- | --- | --- | --- |
| Q4 prose | 86 | 8.547 / 5,176.656 / 23.391 | 2,581.219 | 1,869.957 | 2,469,031,936 / 13,667,614,720 |
| Q4 code | 86 | 8.515 / 5,152.172 / 23.390 | 2,581.332 | 1,845.718 | 2,469,031,936 / 13,667,614,720 |
| Granite prose | 86 | 1.438 / 3,653.141 / 16.219 | 2,581.346 | 355.268 | 1,487,540,224 / 14,649,212,928 |
| Granite code | 86 | 1.407 / 3,655.328 / 16.156 | 2,581.252 | 356.528 | 1,487,540,224 / 14,649,278,464 |

The complete sequence interval is **17,741.438 seconds**; public launcher wall is **17,741.901 seconds**. Totals: 19.907 seconds input loading, 10,325.149 seconds fixed waiting, 4,427.471 seconds native phases and 79.156 seconds case reconstruction. Per-case caps include input loading and reconstruction; all four are below four hours and the sequence is below 16 hours. Native phases separate load/health, tokenization, completion and teardown in the [machine summary](result-summary.json). They exclude waits, identity/source checks and other collector overhead. Equal evaluation counts do not establish lower tuning wall time or human effort. Decode TPS is not CLI or serving throughput.

The separately archived first preflight stopped with zero attempts/native starts and no freeze or case root. Its 23.391-second collector and 23.774386-second supervisor intervals are additional overhead, outside the measured sequence above. A path-spelling prerequisite was corrected after independent audit under the [documented decision](precollection-decision.md); no samples or native budget were reused. See the [zero-start audit](zero-start-preflight-independent-audit.md).

Fresh public read-only validation: **PASS**, exit0, **93.399970 seconds**, four invocation-scoped readers and four model-digest cache entries, **zero additional native calls**. Full live inputs and every retained artifact/record were checked. All63 source files and original 41 frozen files/six history pins/148 native starts remained unchanged. The native report SHA-256 remains `4a74d814daceb07be0d7d3319127e513003dcee9f49f306c4c6fd452745762c9`. See [verification](native-verification.json), [public reconstruction](public-validation.log), [terminal report](native-sequence-report.json), [outer freeze](frozen-sequence-manifest.json), and [corrected raw source archive](corrected-source-snapshot.json).

The [independent final raw-evidence audit](final-independent-audit.md) **passed with no material discrepancies**. It reconstructed all 344 records, 3,440 native artifact hashes, own-reference exact tokens, selections/seeded intervals, owners/waits/memory/cleanup, complete chronology/costs/caps, the 63-file source archive/commit, registration/17 inputs and original 41 files/six history pins/148 records. Its fresh full-weight reads were deliberately deferred to the separate public validation, which passed. See [machine audit](final-independent-audit.json), [reproducible script](final-independent-audit.py), and [separate source-bound arithmetic audit](final-independent-source-bound-auditor.json).

The original Q6 repeatability/transfer proof remains qualified at +9.38% on its held-out workload. These new results establish four completed negative/default-optimal utility comparisons, not broader useful tuning, a gain over manual/already tuned stock, new placement acceleration, universal support or serving performance. The prompts were previously used in other studies, and all cases use one host. Stop this registered search; broader offload/attention/batch controls require their own numerical-scope audit and separate registration before any native experiment.
