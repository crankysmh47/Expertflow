# Real Granite MoE validation

The user authorized broader generalization after closing the Gemma work. Gemma's
bounded PDL experiment is audited NO-GO at34a19b2; its accepted plans are unchanged.

Use exactly the separately downloaded Granite3.1 1B-A400M Instruct Q6_K artifact:
1,099,212,096bytes, SHA4566cfa92be10888026bd3663c83d64e91cd91f874dfb3607596587ff1c8f67f,
publisher revision940d2e1f9f65330615c7c8e980e6c5ac73d3360c. Official config and actual
GGUF metadata agree on24layers/32experts/top8, hidden1024/intermediate512. Inventory
is complete242tensors;72Q6_K routed tensors form three components per layer,
13,762,560bytes each,41,287,680byte bank and1,290,240byte complete expert bundle.
The existing pinned pristine backend supports granitemoe's norm/softmax/SILU MoE
graph and model scaling. No new native build or backend patch is required.

Normalize through a separate Granite adapter and reviewed eligibility provider,
factoring only shared routed-inventory validation. Bind actual architecture,
metadata/tensor shapes, artifact, runtime, host and immutable source objects.
Reject incomplete/altered inventories, other weights/builds, unsupported controls
and capability failures. Profiles/static/cache/fork paths are outside this track.
Legacy compiler-input loading still verifies the configured distinct fork;
only the pristine stock runtime is executed.

The trusted provider declares a GPU-resident baseline:ngl99/CPU-MoE false, threads12,
graphs on, F16KV, batch2048/ub512, ctx4096,512prediction, fixed prompt/seed/temp.
The1.10GB model fits this GPU; actual memory/reserve/ownership remains a live gate.
Choose this baseline explicitly rather than treating Gemma's CPU-MoE policy as
universal. It is not a measured placement comparison or a global-best claim.
Reference settings derive from trusted provider proof; historical Gemma providers
keep their original CPU-MoE default. Product arms bind the source baseline's
CPU-MoE flag while preserving all other pristine controls. Search may vary only
threads and CUDA graphs on the accepted fixed placement; reject PDL/static.

Freeze and commit reviewed code before native retention. Ten own-model reference
processes must have stable tokens, CV<=10%, exact memory/cleanup and source/host
identity. Then twenty fresh paired product processes require CI90 equivalence
within[-2%,2%], one-sided95lower>-2% and bothCVs<=10%. Only after acceptance run
explicit8/12/16threads x graphs on/off, three complete seeded blocks=18screens,
plus at most one independent20-process confirmation for the top challenger.
Confirmation requires gain>=2%, CI95lower>0 and bothCVs<=10%, own reference tokens,
memory/reserve/cleanup/ownership. Incumbent first consumes no self-confirmation.
Maximum68processes across all gates; no retries/discards/retuning or extra finalists.
Keep reference/product/search raw artifacts and source frozen during timing.

Independently reconstruct artifacts, controls, ranking, statistics and published
receipts; perform actual read-only CLI validations. Document actual cross-family
coverage, unsupported scopes and cache invalidation. Real second-family evidence
does not establish universal model/hardware support or a Gemma quality/TPS gain.
