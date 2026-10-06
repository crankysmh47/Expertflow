# Stock utility execution state

Updated 2026-10-04. Measured implementation checkpoint: `0370579`, on `ef-v2`.
The [registered protocol](protocol.md), [source freeze](source-freeze.json), and
[main manifest](main-frozen-manifest.json) preceded native collection.

## Main workload

Terminal state: **PRODUCT-VALIDATION-STOP**. Utility gate **PASS-STOCK-UTILITY**,
from all 86 retained utility processes; final product gate **INCONCLUSIVE**,
from 20 additional fresh runs. The [independent audit](main-audit.json) verified
all 106 native records, artifacts, identities, costs and statistics.

- Both independent 18-evaluation searches selected 12 threads/graphs on.
- Fresh default/automatic means: 20.642049/23.264136 TPS.
- Paired geometric improvement: 12.702875%; bootstrap 95% interval
  [12.452222%, 12.993406%]. Arm CVs: 0.323447%/0.243381%.
- Independent manual/automatic comparison: paired change -0.258933%,
  bootstrap 90% interval [-0.789746%, +0.224924%], within ±2%.
- Search cost: 18 native candidate evaluations each. Audited native phase sums
  740.324s/772.056s are descriptive, not full wall-time or human effort.
- Fresh direct/sealed means: 21.691063/21.950691 TPS; paired change +1.170741%,
  CI90 [+0.133624%, +2.342449%]. The upper bound exceeds the fixed +2% margin.

All native correctness/identity/memory/reserve/cleanup checks passed. No new
accepted product artifact was published; no consumer or transfer run occurred.
There are no remaining model servers. The structural selected-plan JSON in the
raw root is unaccepted and must not be used as an accepted consumer input.

Raw root: `C:/models/expertflow/runs/compiler-stock-utility-main-20261004`.
All 106 attempted launches are retained, with 106 distinct observed owners.
Maximum 107; unused consumer budget cannot retry the failed gate. No retries,
discards, extra finalists or source changes. Historical stock/placement/cache
verdicts stay closed. See [full result](report.md), [validator output](main-validation.log)
and [execution decisions](execution-ledger.md).

## Conditional transfer and product work

The separately committed code prompt uses the same context 4096/predict 512/
seed 42/exact/F16 controls and remains untouched by inference. Its 107-process
budget was conditional on complete main acceptance; that condition was not met.

Current scope remains validated stock selection/reproduction with the narrow
main defaults gain reported separately. Consumer, transfer, CLI expansion and
wider controls are blocked by product acceptance. Next is a separately reviewed
bounded repeatability/acceptance protocol with a concrete new justification,
fixed gates and budget before further native work. Do not resample this study.
