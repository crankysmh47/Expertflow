# Bounded stock search implementation

Execute inline with the implementation checklist under the already authorized
stock configuration discovery goal. Preserve current branch and user edits.
Spec: ../specs/2026-10-04-bounded-stock-search.md.

## 1. Generic scheduling space

Files: src/expertflow/compiler/stock_search.py;
tests/test_compiler_stock_search.py.

Write/observe RED tests before implementation. Add topology_anchors(host, T),
semantic_fingerprint(candidate), scheduling_space(base, host, graph_modes,
excluded_threads). Return immutable candidates/exclusions/untested counts.
Validate complete affinity and supported <=64 logical processor topology;
explicitly reject partial/ambiguous processor-group control. Candidate changes
are limited to WorkloadIR.threads/cuda_graphs and RuntimeSettings.cuda_graphs;
rebuild workload identity and clear old evidence/sealing status. Semantic
fingerprint includes all non-tuning fields and model/hardware/runtime identity.
Test 4/8, 8/16, 16/32, non-SMT and uneven synthetic hosts, static rejection,
incumbent exclusions, semantic input changes and identity invalidation.

## 2. Eligibility and manifest

Add an eligibility provider contract with a Gemma4 Q6_K pinned-runtime provider.
Resolve adapter/quantization/runtime source, verify pristine source object IDs
for inspected CPU and CUDA paths and sanitized graph-optimization environment.
Unknown family/quantization/revision fails before any native launch. Additional
providers require explicit operation-path evidence, never a universal boolean.
Test registered synthetic provider wiring separately from native eligibility.

Manifest records full candidates, incumbent product plan/receipt/database pins,
host, semantic fingerprint, numerical provider/proofs, exclusions, screening
schedule, confirmation schedule/gates, source files/commit, UUID and fresh
boundary. Screening: three seeded complete blocks of four configurations.
Confirmation: ten balanced pairs, independent from screening. Declare maximum
32 new processes and no retries. Canonically hash manifest before launch.

## 3. Execution and reconstruction

Reuse ServerMeasurementRunner and EvidenceStore. New execute_stock_search uses
a fresh output/database and verifies the accepted Stage A prerequisite first.
Import source measurements read-only with preserved IDs. Capture host before
each launch; pass the expected host so the runner writes independent hashed
launch binding. Persist each outcome/row immediately. Verify exact native
tokens against prerequisite, candidate/settings/stage, complete unique process
owners, artifact root/freshness, owned memory/reserve/cleanup and frozen hashes.

Screen ranking uses geometric within-block incumbent ratios. Only first-ranked
challenger is confirmable. Reconstruct from verified native artifacts rather
than report strings; independently recompute ranking and confirmation bootstrap.
If incumbent wins screening, recommend the Stage A plan with explicit screening
uncertainty; otherwise require the new confirmation gate to recommend challenger.
Any partial collection or identity failure stops without recommendation.

Tests: fake native artifact runs, deterministic schedule/budget, rank/tie,
independent confirmation, exact token mismatch, duplicate ownership, changed
host/source/prerequisite, forged ranking/gate/finalist, missing rows, historical
reuse, and atomic no-half-publication. No fake fixture implies live performance.

## 4. CLI, review and live experiment

Add scripts/benchmark_compiler_stock_search.py using existing input loaders and
optional explicit eligibility/source/exclusion arguments. Provide generate-only,
fresh execute and read-only reconstruct/validate actions. Keep legacy compiler
commands unchanged. Add CLI failure tests. Run focused RED/GREEN, full suite,
compileall/diff checks. One independent review/fix pass before native collection.
Commit source/protocol/manifest preparation before retaining measurements.

Run the frozen search once only after Stage A passes. Inspect live process
handles during observation timeouts, retaining all outcomes. Independently
reconstruct and save machine-readable report/receipt and artifact hashes.
Compare old TPS descriptively; do not substitute old rates for current controls.

## 5. Generalization and completion audit

Verify adapter/provider contracts against multiple synthetic model inventories,
topologies and runtime capabilities. Inventory local weights/cache paths and
document which families/quantizations can actually execute. Add a second live
model contract only when supported artifacts and eligibility proof exist; do
all independent interface/documentation work before reporting missing weights
as an external blocker. Document actual live coverage and unsupported hosts.

Audit the complete original goal: usable Stage A stock output, reproducible
current-model search with exact strongest validated recommendation/coverage,
generic model/runtime/host interfaces and cross-family/topology verification.
An inconclusive experiment or unavailable family must remain visible. A passing
contract suite alone cannot satisfy live model or performance requirements.
