# Bounded stock scheduling search

This is the concrete Stage B/C design under the authorized stock discovery plan.
It is not a retained native experiment manifest. Native execution requires the
Stage A product gate to pass, implemented verifier checks, and a separately
frozen manifest of exact candidates, source proofs, host and budgets.

## Reusable contracts

Separate semantic workload (prompt, context, output count, seed, sampling, KV,
batch shape), model/runtime/GPU identity, captured host, tuning space and budget.
Only thread count and graph mode vary in this scheduling subspace. Each concrete
candidate still retains the existing full WorkloadIR and CandidatePlan identity.
Semantic fingerprints remove only those two declared tuning controls. Unknown
controls, static placement, changed precision or changed semantic inputs fail.

An eligibility record names the exact upstream revision, applicable adapter and
quantization, inspected operations and immutable source object IDs. Kernel/source
equivalence does not establish arbitrary model equivalence: unsupported adapters
must provide a reviewed operation-path proof before entering this subspace.
Every retained native run must also match the incumbent's exact prompt/generated
tokens and pass existing owned memory, reserve and cleanup checks.

## Candidate generation and declared coverage

Use physical cores P, logical processors L and incumbent threads T, requiring
1 <= P <= L and a compatible captured affinity. Deterministic topology anchors
are P, P + floor((L-P)/2), L and T, deduplicated and bounded to 1..L.
Cross eligible anchors with graph modes on/off when supported by the pinned
runtime. Explicit exclusions remove anchors before collection; they never count
as measured evidence. Report all omitted thread counts and excluded candidates.

For the current P=8, L=16, T=12 host, retain threads12/16 and graphs on/off:
four candidates, including threads12/graphs-on incumbent. Exclude threads8 using
the prior independent negative result (-7.1788%, CI95[-9.1042,-4.7441]) as a
decision to avoid reopening that hypothesis, not as host-compatible cached
measurement or proof about threads8/graphs-off. All other thread counts remain
untested. Broader explicit spaces are supported by the method, but this run
cannot claim their coverage or global optimality.

## Numerical scope

Inspect pristine upstream a7312ae94f801fc9c6786dc56e38df57b964f697 source,
not merely fork capability strings. CPU directory comparison is unchanged.
For Gemma4 Q6_K on this x86 path, standard complete-output vec_dot and Q8_K
block-local quantization partition work without altering reduction order; see
the existing eight-thread source analysis for operations and line references.
Graph mode switches capture/replay versus direct evaluation in the same node
evaluation function. GGML_CUDA_GRAPH_OPT is disabled in the sanitized launch
environment, so its optional graph rewrite cannot enter this comparison.
Keep model bytes, CPU SIMD binary, quantization, F16 KV, placement and batches
fixed. These inspected conditions admit this model/runtime subspace only;
token guards remain mandatory. No arbitrary GPU placement or KV tuning enters.

## Fixed measurement and selection

Screen exactly three shuffled complete blocks, each containing all four
candidates once: twelve cold processes. Shuffle with seed20261004. Freeze the
full order before any launch. Rank candidates by geometric mean of their TPS
ratios to the same block's incumbent. Deterministic ties favor the incumbent,
then candidate identity. Report every run and screening uncertainty; screening
rates cannot publish a new recommended plan.

If a non-incumbent screens first, collect an independent ten balanced pairs
(twenty cold processes) against the incumbent, with seed20261003 and the existing
10000 whole-pair log-ratio bootstrap. Acceptance requires point gain >=2%,
two-sided95% lower bound >0, both CVs <=10%, exact tokens and all ownership,
memory/cleanup/host gates. Otherwise retain the independently Stage A-validated
incumbent and the full negative/inconclusive confirmation evidence. No second
finalist, extended budget, retry or discarded outlier. If incumbent screens
first, retain it without an unnecessary self-comparison.

Maximum budget is32 new processes; screen-only budget12 if incumbent wins.
Any collection/identity/token/memory/cleanup failure stops, with partial evidence
retained and no discovery publication. Source/host changes invalidate reuse.
Independent reconstruction binds run artifacts, manifest, candidate/stage IDs,
unique process owners, fresh collection boundary and computed selection/stats.

## Output and generalization

Publish a manifest/report and evidence-backed recommendation receipt. Identify
the strongest validated recommendation in the declared measured space, with
screening rank, confirmation status, exclusions and uncertainty. Do not claim
the true population maximum or historical speed recovery without evidence.
Historical comparisons remain descriptive and separate from paired controls.

Provide CLI commands to generate/freeze a space, execute once, reconstruct and
validate a recommendation using existing adapter/runtime inputs. Test changed
model, quantization, runtime, driver, host topology, affinity and power settings
invalidate reuse; test multiple synthetic MoE families/topologies explicitly
as contract tests. Inventory real local weights and supported adapters before
requesting downloads. Missing second-family support must be documented after
independent reusable-method work is complete.
