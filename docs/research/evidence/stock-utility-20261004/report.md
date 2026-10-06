# Stock utility result

Terminal result: **PRODUCT-VALIDATION-STOP**. The stock-tuning utility comparison
passed, but fresh sealed-plan acceptance was **INCONCLUSIVE**. Preserve this
partial result; no consumer, transfer, retry, discarded sample or extra finalist.

Measured source `0370579` on `ef-v2`. The
[placement audit](../placement-proof-20261004/feasibility.md) found no new exact
mechanism in the pinned kernels and activated this fallback. The
[registered protocol](protocol.md) and both prompts preceded native collection.

## What was established

Both independent 18-evaluation grids selected **12 threads/graphs on**, among
8/12/16 threads crossed with graphs on/off, on the pinned Gemma Q6/Windows/RTX 5060 Ti
host. Defaults resolve to 8 threads/graphs on, with CPU-MoE fit and all workload/
placement/precision controls fixed. This is a defaults comparison within that
declared scope, not a gain over strongest manually tuned stock.

| Fresh comparison | Mean TPS: control/automatic | Paired change | Confidence interval | Verdict |
| --- | --- | --- | --- | --- |
| Resolved defaults vs automatic | 20.642049 / 23.264136 | +12.702875% | 95%: [+12.452222%, +12.993406%] | Utility gain PASS |
| Independent manual vs automatic | 22.256768 / 22.199771 | -0.258933% | 90%: [-0.789746%, +0.224924%] | Manual equivalence PASS |
| Direct vs sealed selected plan | 21.691063 / 21.950691 | +1.170741% | 90%: [+0.133624%, +2.342449%] | Product equivalence INCONCLUSIVE |

All comparisons use ten balanced fresh pairs, 10,000 whole-pair bootstrap draws,
seed 20261003 and the registered nearest-rank empirical percentiles. Utility arm
CVs were 0.323447%/0.243381%; manual comparison CVs 1.917465%/2.083848%; product
CVs 0.637605%/2.507892%. All variance gates passed.

Product noninferiority passed, but the 90% upper bound exceeded the unchanged
+2% equivalence limit. The sealed point estimate was faster; this stop does not
show degraded execution or a numerical failure. It does prevent new accepted
publication and the conditional consumer/transfer stages. The underlying direct
and sealed configuration identities/settings matched. Performance variation's
cause remains unestablished; no native diagnostic experiment was added.

## Cost, correctness and retention

Automatic and manual each used 18 native candidate evaluations. Their observed
native load/tokenize/completion/teardown sums were 740.324s and 772.056s. This is a
descriptive phase-time observation, not total pipeline time or a lower wall-time/
human-effort claim. Reference, confirmation and product-validation overhead is
retained separately. Existing historical rates are not contemporary controls.

Exactly **106 of 107 permitted native processes**: reference 10, automatic 18,
manual 18, default/automatic 20, manual/automatic 20, product 20. Every attempt
started one observed native process; all 106 owners were distinct and retained.
All native prompt/generated IDs, actual runtime/arguments, memory/reserve and
cleanup checks passed. Maximum process-owned VRAM 3136.660 MiB; minimum observed
device free memory 10556.027 MiB. No model server remains running.

The one consumer process and the fresh 107-process transfer budget were unused
after the product stop. The committed held-out prompt remains unmeasured.

## Audit and artifacts

- [Main manifest](main-frozen-manifest.json), [source freeze](source-freeze.json).
- [Main report](main-report.json), [product report](main-product-report.json).
- [Independent raw audit](main-audit.json) and [standalone audit code](independent_audit.py).
- [Actual validator output](main-validation.log) reconstructs the terminal stop;
  its nonzero exit is expected, not a successful product acceptance.
- [Implementation verification](implementation-verification.json): 767 tests pass,
  7 optional historical source skips, 6 applicable pinned native source checks;
  all 3 fresh review findings reproduced/fixed before collection.
- [Execution ledger](execution-ledger.md): completed work, review fixes, rulings
  and their costs; [final verification](final-verification.json).

Raw roots/databases remain under
`C:/models/expertflow/runs/compiler-stock-utility-main-20261004`. Read-only
reconstruction and the independent raw audit verify
all 106 native records/artifact hashes, original context/source/host, retained
process identities and independently recomputes timings, costs and statistics.
The source/host guards remained unchanged throughout collection. Historical
accepted plans, quality/cache/thread no-go results and all six history pins match.

## Product decision and next action

Current product scope stays **validated configuration selection/reproduction**.
The declared main-workload tuning gain is confirmed, but end-to-end new tuning
product acceptance and transfer are unproven. Do not call this a new accepted
MoE placement accelerator, global stock optimum or universal automatic gain.

CLI/product expansion and wider controls remain deferred until complete proof.
Next work is a separately reviewed, bounded repeatability/acceptance protocol
based on this retained variation, with a concrete new justification and fixed
budget before any further native run. It cannot relabel this106-process study,
waive its equivalence gate, or simply resample until passing.
