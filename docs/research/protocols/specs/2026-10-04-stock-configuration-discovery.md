# Quality-preserving stock configuration discovery

The active user goal is to follow the replay-validation plan, then establish a
way to find the best stock configuration for the current Gemma4 Q6 model and
hardware, and generalize the method to MoE model/hardware configurations. This
is an execution authorization. Preserve ef-v2, unrelated user files, all prior
negative evidence and original builds. Do not alter numerical kernels, quality
policy, quantization, model bytes, context, prompt or output length to manufacture
a speed improvement. No merge, push or external publication.

## Historical speed audit and scope

Pristine historical stock: 22.9667 TPS over three512token CLI runs; original
compiler confirmation:24.411TPS over ten server processes and replay25.383TPS.
The latter failed its predeclared absolute2% replay gate despite being faster.
These are different experiments and interfaces, not matched proof of a speed
regression. Static CPU-to-CUDA result28.13TPS is quality-ineligible; its PPL
upper confidence bound failed. Four-slot35.6699TPS is aggregate concurrency
throughput with nondeterministic outputs, not exact single-request decode speed.
Retain these facts and their source hashes in a machine-readable audit.

The method must report the strongest **measured eligible configuration in its
declared search space**, coverage, exclusions, remaining uncertainty and budget.
Never claim global optimality from a small sweep or universal support from one
Gemma experiment. Historical25TPS is a comparison target, not a mandatory claim
obtainable by selective retries or by weakening quality.

## Stage A: new stock-product replay protocol

Keep the original Phase3 validator and original VALIDATION-STOP unchanged.
Add an explicitly versioned paired-stock-product-v1 protocol. Start from the
verified diagnostic stock plan and pristine runtime. Exactly ten balanced
direct/sealed pairs (twenty fresh cold processes), seed20261003, five orders
each. Bootstrap10000 whole pairs with the existing log-ratio implementation.
Require90% equivalence interval strictly inside[-2%,+2%], one-sided95% lower
bound>-2%, both CVs<=10%, all exact prompt/generated tokens against the verified
source reference, owned memory/reserve and cleanup, unique process identities,
matching candidate/settings/model/runtime/GPU/host identities. No native retries,
discarded samples, optional extensions or old A/A relabeling as product evidence.

On success, seal a fresh stock plan using the twenty new verified measurements
and write a sidecar acceptance receipt binding plan hash, protocol version,
source-plan snapshot/hash, frozen experiment, twenty pair/arm measurement IDs,
CPU/OS/power environment and independently reconstructed statistics. Consumers
must reverify the receipt against EvidenceStore; a PASS string or boolean is
insufficient. Old measurement-refinement A/A stages are ineligible. Failure or
inconclusive means no published plan/receipt. Source plan/DB remain unchanged.
A source-plan snapshot can change only its evidence IDs/sealing metadata when
published; launch identities/settings must remain identical to those tested.

Validate/run gain optional acceptance-receipt support. Existing calls retain
legacy behavior. Accepted-plan execution verifies receipt, host and pinned
runtime, then measures a fresh owned run with exact token/memory/cleanup checks;
it does not turn one noisy rate into a new paired-validation verdict.

## Stage B: strongest-stock search for this model

After StageA passes, implement a reusable search-space contract, deterministic
candidate generation, explicit numerical eligibility and budgeted measurement.
Derive CPU ranges from topology, memory constraints from model/KV inventory and
VRAM, backend controls from pinned runtime capabilities. Every candidate must
carry a complete identity and semantic workload fingerprint independent of
tuning knobs. Compare rates only within the same semantic workload and host.
Historical rejected configurations remain visible; reuse only fully verified
evidence with compatible runtime/workload/environment, never unqualified TPS.

First subspace: stock CPU expert scheduling and CUDA graph orchestration on the
unchanged Q6 arithmetic path. Inspect actual source before certifying a setting
class. Thread range is not hardcoded to this machine. Context/prompt/predict/KV,
quantization, runtime CPU SIMD flags and numerical placement remain frozen.
The search manifest declares the exact evaluated thread counts/graph controls,
the method for choosing them, excluded or untested counts and finite native
budget before collection. No repeated trials of the already rejected prefetch
kernel or caching path. Screening selects a finalist, then independent balanced
confirmation against incumbent establishes gain or noninferiority. Publish only
the independently validated winner, never a screening maximum. A first screening
gain does not justify an unbounded sweep.

Other stock knobs (offload/expert placement, batching, KV formats) require their
own numerical eligibility contract; runtime availability is not an exactness
proof. Exclusions must state the evidence needed to admit them. Preserve exact
output requirements and prohibit the historical quality-ineligible static mode.

## Stage C: reusable MoE/hardware method

Separate model metadata/adapters, runtime capabilities, host identity, search
space, budget and objective from this Gemma descriptor. Provide explicit plugin
contracts and capability rejection for unsupported families/settings, candidate
cache invalidation across model/quantization/topology/driver/runtime/power policy,
and tests on multiple synthetic inventories/topologies without claiming those
are live model measurements. Inventory locally available models; do not download
large models without a concrete need and budget. Add a second live adapter and
validation when suitable weights are available; missing weights become a real
external blocker only after all independent method work is completed.

Completion requires StageA live-valid stock output, StageB reproducible search
and declared strongest exact stock configuration for this host/model, StageC
reusable interfaces and cross-topology/family tests with explicit live coverage.
No benchmark method can promise one configuration works best for every workload.
