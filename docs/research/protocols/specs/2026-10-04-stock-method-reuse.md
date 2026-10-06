# Stock method reuse and independent Q4 validation

This extends Stage C of the approved stock-discovery specification. It does not
alter the frozen Q6 experiment or turn Q4 into a Q6 quality-preserving speedup.
Implement and collect only after the Q6 search is terminal and independently
audited. Stay on ef-v2 and preserve unrelated user edits and native builds.

## Required reusable behavior

The reusable entry points take model descriptor, verified inventory, hardware,
semantic workload, pristine runtime, host snapshot, reviewed numerical provider,
explicit search coverage and finite budget. A model adapter accounts for bytes
and architecture; it does not certify numerical equivalence. An eligibility
provider is reviewed code registered by family and quantization; a manifest must
not select or execute arbitrary provider code. Providers bind actual weights,
runtime binaries/build flags, immutable operation source objects and supported
host architecture. All native launches bind the full frozen experiment and host.

The current topology anchors are physical cores, midpoint to logical cores,
logical cores and incumbent threads. These are declared coverage, not an
exhaustive optimum. Partial affinity, ambiguous multi-socket or processor-group
topologies fail explicitly. A caller on another supported topology supplies an
explicit space config and exact budget; the Gemma Q6 threads8 exclusion must
never leak into another model or host. Unsupported settings list the missing
operation proof and validation needed to admit them.

Identity changes invalidate reuse: model bytes/quantization/inventory, semantic
workload, runtime/CPU SIMD/dependencies/CUDA runtime, GPU/driver, CPU topology,
affinity, RAM/OS/thread environment or complete power policy. No portable TPS
claim comes from synthetic fixtures. Record live and contract-only coverage
separately. Recommendations are strongest validated in the declared space;
neither universal support nor global optimality follows from a bounded search.

Provide accepted recommendation execution as well as read-only validation.
Reconstruct the search receipt, compare actual inputs to the frozen semantic
workload, then launch the published candidate's tested settings and verify fresh
tokens, memory/reserve, cleanup, owner uniqueness and host stability. Report a
fresh execution measurement without turning its single TPS into a new acceptance
verdict. Do not add an extra native run to either experiment's fixed search
budget; test this consumer with native artifact fixtures and expose an explicit
execution action for subsequent use.

## Second real artifact

Local Q4 file:
`C:/models/expertflow/google--gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf`.
Header inspection confirms 14,439,361,440 bytes, Gemma4, 30 routed layers,
128 experts/top8, sixty Q4_0 routed weight banks and thirty F32 expert scales.
658 total tensors include one Q6_K output weight. Recompute the full file digest
after Q6 collection; the historical digest is a comparison, not verification.
Generate a complete tensor inventory from the actual header and normalize it
through the existing Gemma4 adapter with a separate Q4 descriptor/runtime input.

Q4 source proof must inspect its Q4_0/Q8_0 dot/conversion path and any AVX2
repacking dispatch, including prompt evaluation. It must show that admitted
thread/graph controls partition complete outputs while preserving each output's
arithmetic path. Do not infer this from the Q6 proof or filename. Native prompt
and generated tokens must match a fresh Q4 reference for every retained run.
Failure means Q4 remains ineligible; retain the negative evidence and complete
other independent method work. No quantization quality comparison to Q6 is made.

## Frozen Q4 budget and gates

Only after the operation audit admits this scope, use unchanged pristine build,
CPU-MoE/ngl99, F16 KV, context4096, batch2048/microbatch512, prompt/predict512,
temperature0/seed42 and incumbent threads12/graphs on. Reuse the current host
policy without tuning it. Freeze all Q4 inputs, relevant provider/driver/compiler
source, protocol, environment and paths before collecting. No retries, partial
reuse, discarded runs, alternate incumbent or gate changes after results.

1. Ten fresh reference processes at the single incumbent configuration. Verify
   every native artifact, exact token stability, owned memory/reserve, cleanup,
   unique ownership and CV<=10%. Seal a diagnostic source plan from these ten
   records. This is a stable reference, not strongest-stock evidence or replay
   acceptance. A bounded reference collector must reconstruct this gate.
2. Twenty fresh processes using the existing paired-stock-product-v1 protocol.
   Its original equivalence/statistical/correctness gates remain unchanged.
   Publish only on PASS-STOCK-FALLBACK; otherwise stop Q4 collection.
3. An explicit Q4 search with no thread exclusions: threads8/12/16, graphs
   on/off, three complete screening blocks (18 processes), optional single
   finalist confirmation (20 processes). Maximum search38, total Q4 maximum68.
   Exact tokens and all existing search gates apply. A challenger requires
   point gain>=2%, CI95 lower>0 and CVs<=10%; otherwise retain the validated
   incumbent. Screening incumbent consumes no self-confirmation.

Independent read-only audit and CLI verification must reconstruct the reference,
product receipt and search recommendation. Report actual consumption and all
exclusions. Stop on identity, correctness, memory, cleanup or environment
failure; keep raw evidence and verdict. No download or second-family claim.

## Historical source preservation

Q6 manifests bind the entire compiler source bundle at f22bb54. Future compiler
changes must not silently reinterpret that receipt. Preserve a checkout of its
frozen revision for historical reconstruction and document the exact validation
command. A new compiler version needs explicit compatibility verification or
fresh native validation before claiming accepted execution under the new code.

## Completion evidence

Require audited terminal Q6 coverage, reusable contracts and CLI instructions,
identity-invalidation tests across multiple synthetic topologies/families,
verified Q4 normalization/source scope and its independent live validation,
honest capability limits and a requirement-by-requirement audit of the original
Stage A/B/C objective. Q4 is a second quantization of one real family. Supporting
another family requires its adapter, immutable operation proof and own live
validation; this method must expose that extension boundary instead of claiming
unmeasured compatibility.
