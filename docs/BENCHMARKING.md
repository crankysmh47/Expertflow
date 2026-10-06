# Measure a local profile

Create a working baseline with `local setup`, then close or idle competing GPU
work yourself. The collector checks for three consecutive idle utilization samples
within five seconds. It does not close applications. Every job needs a fresh
output directory and an absolute wall budget; failures retain diagnostics.

```powershell
expertflow local bench --profile profile.json --budget-seconds 120 --output-dir bench-01
expertflow local verify-job --job-dir bench-01
expertflow local support --job-dir bench-01 --output support.json
```

A default benchmark uses one owned process, a 16-token warmup and three
256-token completions on a fixed public prompt. Reports include decode tokens per
second, observed streaming first-token latency, complete token/timing receipts,
and memory samples after healthy load. Memory is a sampled resident peak, not a
load-allocation maximum. Decode speed is not total request latency or serving capacity.

## Reviewed exact tuning

```powershell
expertflow local tune --profile profile.json --budget-seconds 600 --output-dir tune-01
```

Only the bundled audited Gemma Q6/Q4 and Granite Q6 identities on the pinned
Windows runtime qualify. Setup must use upstream thread defaults; create a fresh
profile without `--threads` if needed. Tuning preserves placement, arithmetic,
quantization, context and F16 KV, searching threads 8, 12 and 16. It permits at
most 14 native launches with 30-second spacing and never extends a closed budget.

An apparent candidate proceeds to two held-out prompts in five alternating pairs.
Token identity and host/workload guards must pass; paired variance above 10% is
inconclusive. Selection requires the lower 95% paired bootstrap gain bound above
5%. Successful selection writes a separate `selected-profile.json`; input profiles
are never overwritten. Break-even estimates cover comparable decode savings only.

| Outcome | Meaning and next action |
| --- | --- |
| MEASURED | Baseline observation; reuse the working profile. |
| VERIFIED-IMPROVEMENT | Confirmation passed within the stated contract; inspect the selected profile. |
| NO-MEASURABLE-GAIN | Completed comparison did not find an accepted improvement; keep the baseline. |
| INCONCLUSIVE | Budget, variance, token or environment guard prevented a conclusion; keep the baseline. |
| UNSUPPORTED | No reviewed tuning contract; use untuned run/serve/bench. |

The first Granite search ended `NO-MEASURABLE-GAIN` after four processes and
124.89 seconds. See [native receipts](evidence/local-product-20261006/report.md).
There is no speedup guarantee, global-optimum claim or quality-changing search.
