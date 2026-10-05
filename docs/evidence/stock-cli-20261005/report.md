# Stock CLI, reader reuse and wider registration

The three requested product stages are implemented separately from the closed
scientific studies. All original compiler/collector files remain byte-for-byte
unchanged. The public adapter lives in `src/expertflow/stock/`; adding it does
not alter the historical compiler source glob or bypass any source-map guard.

## Real read-only validation

`expertflow stock repeatability validate` reconstructed the existing 148-call
study with **PASS-STOCK-REPEATABILITY-TRANSFER**, exit0, in **207.944475 seconds**.
The prior exact legacy CLI invocation took 2,191.483952 seconds. This is a
single descriptive same-host comparison, approximately 10.54 times shorter;
it is not a controlled estimate of speedup on other evidence sets or hosts.
No model was launched. Native-start inventory stayed148, report bytes stayed
unchanged, all41 frozen source/prerequisite files and six historical pins
matched before and after. [Raw public output](public-validation.log),
[timing and integrity record](validation.json), [verification helper](validate_live.py).

After the final reporting fixes, a fresh read-only invocation passed in
**221.152227 seconds**, exit0, again with five readers, all41 frozen files/six
history pins unchanged and zero new native calls. This invocation ran alongside
the final CPU test suite; both timings remain descriptive. Its
[raw final output](final-public-validation.log) and [final verification](final-validation.json)
bind the exact adapter source hashes and preserve the first timing record.

Previously, the repeatability validator constructed a new EvidenceStore for
each attempt and consequently lost that reader's model digest cache. The new
adapter supplies one existing EvidenceStore per resolved database during a
validation invocation: five stores/five model cache entries for this study.
The full digest still runs when a store first sees a model or its size/mtime/
ctime fingerprint changes. Input loaders and the independent prerequisite
auditor still perform their own verification; five cache entries are not a
count of every full-weight read in the entire command.

Every measurement still rechecks native artifacts, runtime/dependencies,
settings/argv/environment, tokens, owned process, timing, memory and cleanup.
No verified row, artifact or statistical verdict is cached. Constructor aliases,
import paths, working directory and module state restore after delegation.
Caches survive only that synchronous invocation. This retains the original
stat-based cache contract; it does not add protection against falsified file
metadata or turn read-only validation into a new native performance result.

## Preserved measured source

[measured-source-snapshot.zip](measured-source-snapshot.zip) retains exact bytes
of all41 bound source/prerequisite files, including the original selected plan,
closed report and source database. [source-snapshot.json](source-snapshot.json)
lists each original absolute path, archive member, size and digest, plus the
measured source commit. Every member was rehashed after archiving. This archive
preserves provenance; it does not relocate native artifact bindings or make
the local model/runtime/database evidence portable. Original source maps,
protocols, gates, receipts and negative studies remain unchanged.

## Public workflow and costs

The public CLI routes reference generate/run/validate, product run, search
generate/run/validate/execute and closed utility/repeatability validation to
their existing project drivers. It rejects experiment/action overrides,
including argparse abbreviations. Native behavior is unchanged; fresh output,
identity, eligibility and budget checks remain in those drivers.

Installed help works anywhere. Execution requires an explicit matching research
checkout and its original inputs; compiler bytes in the package must match the
project's source map. A separately installed wheel passed six startup/help/
missing-checkout/registration checks outside this checkout, without loading a
model. [Packaging record](packaging.json), [smoke helper](verify_package.py).
The final rebuilt wheel separately passed the same six checks;
[final packaging record](packaging-final.json).

JSON output distinguishes reconstructed negative/neutral gate outcomes from
identity/environment invalidation. It includes covered model/workload/runtime/
host identities, available collector/search-phase costs and original statistics.
Default-optimal selection is explicit. Neither reproducible selection nor
equal18/18 native evaluation cost establishes a global optimum, a gain over
already tuned/manual stock, reduced human effort or serving throughput.

## Wider tests

[Registration](../../../configs/compiler/stock-coverage-20261005.json) and
[protocol](../../superpowers/specs/2026-10-05-stock-coverage.md) pin four cases:
Gemma Q4 and Granite Q6, each with prose/code prompts. Maximum107 calls per
case/428 total; fixed30s spacing and4hours per case. Preserve the original
5% defaults-gain, positive95% lower bound, manual/product ±2% equivalence,
CV≤10% and exact numerical/native correctness gates. No case pooling/retries.

The coverage inspector rehashes small input/protocol pins and recomputes
normalized metadata, six candidate identities and seeded schedules. It reports
**REGISTERED-NOT-RUN**, `execution_ready: false` and `weight_hashes_verified: false`.
It does not rehash GGUF weights or substitute for precollection live identity
checks. Next is a separate reviewed wider collector and immutable source/input
freeze, then registered collection and independent raw auditing. No new utility
result, untouched-workload result or other-host coverage is claimed.

## Verification

Reader lifetime/model-replacement/artifact-tampering controls passed, including
full completed-study fixture reconstruction with five readers. Public routing,
closed-study guards, override rejection, negative-result reporting, registration
mutation controls and installed-package checks passed. Existing neutral/tie and
incomplete-grid controls remain quantitative contract evidence, not native gain.
Focused compiler/CLI/evidence/documentation checks passed94 tests after the
final reporting controls. The [independent review](review.md) found two important
reporting issues; all six regressions failed before the one fix pass and passed
afterwards. Reader guards/source preservation were independently checked.
[Saved real report-schema controls](report-shape-controls.json) confirm product,
reference and search scope extraction without claiming new native validation.
Final full suite: **835 passed, 7 optional historical source-environment skips**
in 1,025.99 seconds. Six applicable pinned native source checks passed separately;
the rebuilt wheel passed six smoke checks. [Final implementation verification](implementation-verification.json)
binds the adapter sources and archived evidence. The previous scientific
implementation checks remain in the original evidence directory and are not
overwritten.

Cleanup exception: automatic approval review rejected deletion of this phase's
ignored scratch directory, stating "blocked by policy". The required evidence
is archived here; the scratch directory is retained without an alternate
deletion attempt. This has no effect on the validation or registration result.
