# Final independent raw-evidence audit

**PASS — no material discrepancies found.** Audited terminal sequence at executed commit `78ad5f0063f2bc372a528e37ec1f45d8be1ceb11`. This is an evidence audit, not an implementation re-review.

All four cases reconstruct as **NO-UTILITY-GAIN**, with86 valid utility records each and344 total attempts/unique native starts. No conditional product, consumer, selected-plan artifact, extra database, or unaccounted native start exists. Small positive Q4 changes do not meet the preregistered minimum5percent utility gain; Granite remains default-selected. No gate was relaxed and no pooling or family-wide gain is claimed.

| Case | Automatic threads/graphs | Manual threads/graphs | Defaults geometric change and CI95 (%) | Manual equivalence CI90 (%) | Full case seconds |
| --- | --- | --- | --- | --- | --- |
| gemma4-q4-prose | 12/on | 12/on | +1.0295% [+0.8820, +1.1931] | [-1.0142, +0.1114] | 5208.609 |
| gemma4-q4-code | 12/on | 12/on | +0.7818% [+0.6263, +0.9333] | [-0.2011, +0.0827] | 5184.077 |
| granite-q6-prose | 8/on | 16/on | -0.0243% [-0.1483, +0.0863] | [-0.1183, +0.0355] | 3670.813 |
| granite-q6-code | 8/on | 16/on | -0.2350% [-0.3823, -0.0824] | [-0.2013, +0.0387] | 3672.907 |

## Independently verified

- Read all344 SQLite measurement records in exact sequence order, using SQLite mode=ro/query_only and integrity/foreign-key checks. Rehashed3440 bound native artifacts; checked measurement/payload/key/owner hashes, artifact and validation tables, report rows/outcomes, exact roots/databases/stages, launch controls and outer manifest contexts.
- Compared every case's complete prompt/output token IDs against its own reference; every request produced512 output tokens. Verified distinct owners/run IDs/measurement IDs across all344 calls and disjoint owners from the original148-call study. Native observations, process-owned memory, device-free reserve, owned termination, zero exit, settled teardown and no forced kills all pass.
- Recomputed automatic and manual selections from their separate18-record grids, ten-reference CV, paired raw rates, mean/log-ratio values, seeded10000-resample nearest-rank bootstrap intervals and stop reasons. All reference/paired CV gates and manual equivalence gates pass; every defaults gain is below5percent. Independent arithmetic used the source-bound independent auditor, not the collector's evaluate_utility or public wider validator.
- Reconstructed all scheduled waits and phase/load/collection/reconstruction costs from raw evidence. Every prelaunch wait is at least30seconds; actual waits total10325.149seconds versus the10320-second minimum for344 calls. Whole-sequence elapsed17741.438seconds matches its monotonic endpoints; supervisor elapsed17741.901247seconds/exit0 is consistent. All per-call180/300-second limits, four-hour case caps including load/reconstruction, sixteen-hour sequence cap, and case/sequence chronology pass. Phase costs and memory extrema match the public saved results. Mandatory sample interval is0.2seconds; observed maximum gap is approximately0.235seconds.
- Verified all63 exact archived raw source/input files, current freeze inventory, archive digest, and commit78ad5f0 blobs. Three differences are only declared LF/CRLF equivalence. Verified unchanged canonical registration SHA0376ac4d7c5e146bc594f5c22a4b1be803b446ced4c3d9d04a4cb0737af55603 and all17 registered input hashes. Rebuilt model/workload/candidate scopes from pinned small inputs; checked live host, eligibility/runtime inventories and upstream source/default proofs.
- Original41 frozen files and six historical pins remain byte-identical. The old study retains148 measured attempts/start records with matching process identities; original report SHA remains464defd327d43c4f510f3768266d1dacaefc77d25c239afefef5cc4226c1d0c5. All terminal reports/manifests/databases and pinned source hashes remained unchanged during this audit.

## Scope and limits

The audit launched no native process, changed no Git state or pinned file, and ran no full suite. The source-bound independent auditor was invoked with explicit audit-only adapters: read-only SQLite, registered model identity/size/stat checks in place of fresh full-weight hashing, and once-per-binding runtime digests with unchanged stat guards. Raw artifact verification, native-record reconstruction and independent bootstrap arithmetic remained active. Fresh complete model hashing belongs to the root agent's separate public read-only validation. This audit does not claim an independent external profiler measurement or a simultaneous family-wide performance guarantee.

The earlier zero-start preflight archive remains separate; these344 calls belong only to the corrected terminal sequence.

## Reproduction and outputs

Run the archived script from the matching frozen checkout with the unchanged local native roots. Its two launcher inputs are archived in this directory; restore those copies into the owned ignored workspace first if needed:

```powershell
$auditWorkspace = '.superpowers/sdd/2026-10-05-wider-stock-collector'
New-Item -ItemType Directory -Force -Path $auditWorkspace | Out-Null
Copy-Item -LiteralPath 'docs/evidence/stock-coverage-20261005/native-supervisor.json' -Destination "$auditWorkspace/native-job.json"
Copy-Item -LiteralPath 'docs/evidence/stock-coverage-20261005/native-collection.log' -Destination "$auditWorkspace/native-collection.log"
uv run --no-sync python docs/evidence/stock-coverage-20261005/final-independent-audit.py
```

The script writes only owned ignored audit outputs. `final-independent-audit.json` contains complete per-case intervals, controls, reference/token hashes, phase costs, memory extrema, caps and provenance; `final-independent-source-bound-auditor.json` preserves the independent arithmetic auditor's result. An initial audit-helper comparison rejected Python sets as unsupported canonical values before touching native evidence; that local helper was corrected, and the final complete audit passed. The audited code is preserved; one terminal blank line was removed from the durable script for the whitespace check. The byte-exact original remains in implementation-workspace.zip. These reproduction instructions were adapted to durable paths.
