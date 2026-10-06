# Stock method reuse implementation plan

Use the implementation checklist. Spec:
`docs/superpowers/specs/2026-10-04-stock-method-reuse.md`.

Do not change measured compiler/driver/spec while Q6 search is live. Q4 maximum
68 native processes, conditional on its source eligibility and successive gates.
No native retries or altered gates. Preserve ef-v2 and unrelated user files.

## Task 1: finish and preserve Q6 evidence

- [x] Observe existing exec session43284 to terminal; no replacement process.
- [x] Run docs/evidence/stock-discovery-20261004/verify_search.py and the existing
  search CLI validate against its actual database/recommendation.
- [x] Copy recommendation metadata unchanged; record coverage, uncertainty,
  gates, consumption and separate historical speed comparison in report.md.
- [x] Preserve a frozen source checkout for future historical reconstruction.

## Task 2: normalize and audit actual Q4 artifact

- [x] After Q6 timing ends, rehash Q4 bytes; generate complete GGUF inventory and
  separate descriptor/runtime input without changing native build or workload.
- [x] Normalize actual inventory; save ModelIR/hash and tensor accounting.
- [x] Inspect immutable Q4_0/Q8_0 conversion/dot/repack and graph paths; record
  complete source objects, dispatch limits and numerical scope.
- [x] TDD a separate reviewed provider; reject changed weights/build/source/host
  and unsupported controls. Register reviewed code only after proof passes.

## Task 3: reusable fresh reference collector

- [x] Add a generic bounded incumbent reference collector using CandidatePlan,
  CompilerInputs, EvidenceStore and existing ServerMeasurementRunner.
- [x] Freeze all source/provider/driver/protocol/input/host identities and ten-run
  schedule before native collection; hashed launches bind that manifest.
- [x] TDD completeness/order/ownership/freshness/token/CV/artifact reconstruction
  and fail-closed publication. Do not import old summaries as native reference.
- [x] Add CLI support for explicit model inputs and budget; default is read-only
  preparation, native execution requires the explicit run action.

## Task 4: reuse contracts and verification

- [x] Exercise separate inventories/quantizations, topology anchors and explicit
  spaces. Test invalidation across weights/semantic workload/build/driver/CPU/
  affinity/RAM/OS/power/environment and rejected unsupported capabilities.
- [x] Bind any new provider/driver source in search manifests without weakening
  trusted re-attestation. Keep frozen Q6 validation runnable at its revision.
- [x] Add evidence-backed accepted recommendation execution and an explicit CLI
  action. Test actual-input/token/host/ownership failures and tested candidate
  settings; do not append native runs to the frozen search budget.
- [x] Run targeted and full checks, one required independent review/fix pass,
  then commit before Q4 retention. No overlapping tests and native timing.

## Task 5: independent Q4 live validation

- [x] One fresh ten-process reference, audit and diagnostic plan.
- [x] One fresh twenty-process product experiment, unchanged acceptance gates.
- [x] Only after product PASS, explicit six-configuration search,18 screening
  plus optional20 confirmation; publish only independently validated result.
- [x] Independent artifact/statistical/source audit and live read-only CLI
  verification. Record failures without retries or unmeasured speed claims.

## Task 6: method documentation and full objective audit

- [x] Document runnable inspect/prepare/run/validate commands, plugin contracts,
  declared coverage/budgets, capability rejection and cache invalidation.
- [x] Distinguish Q6/Q4 live coverage from synthetic cross-family/topology tests.
- [x] Audit original Stage A/B/C requirements and historical speed comparison;
  do not mark the goal complete while required evidence remains missing.

Final read-only audit: VERIFIED-REGISTERED-STOCK-METHOD,100 distinct native
processes/artifact sets reverified at their frozen source revisions. Q4 consumed
48/68; incumbent ranked first, so no20-process self-confirmation. See
`docs/evidence/stock-discovery-20261004/method-verification.json` and the objective
audit for requirement evidence and explicit historical/universal coverage limits.
