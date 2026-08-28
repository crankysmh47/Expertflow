# ExpertFlow llama.cpp inference compiler design

Date: 2026-08-28

Status: approved architecture
Primary objective: maximum single-request decode TPS

## 1. Purpose

ExpertFlow will become a hardware-aware inference-plan compiler for GGUF MoE
models running on one NVIDIA CUDA GPU through a maintained, pinned ExpertFlow
fork of llama.cpp.

The compiler does not translate model source code. It analyzes a concrete
model, runtime, GPU, workload, and evidence set; searches compatible runtime
optimizations; measures finalists; and emits a versioned execution plan that
the pinned runtime can validate and execute.

The first validated model is Gemma 4 26B A4B. The compiler core remains
architecture-generic for MoE models so that Qwen and additional Gemma variants
can be added through model-family adapters rather than duplicated optimization
logic.

AirLLM-style disk or layer streaming is explicitly out of scope. CPU, Vulkan,
multi-GPU, and non-llama.cpp runtime optimization are also out of scope for the
first release.

## 2. Product principles

1. **Measured decisions.** Generic hardware assumptions may generate
   candidates, but only machine-specific measurements can select a winning
   plan.
2. **Exact by default.** Exact profiles preserve the declared token and runtime
   identity contracts. Quality-changing optimizations require explicit
   approximate mode and separate evidence.
3. **Strongest stock floor.** Every optimization is compared with the strongest
   compatible stock llama.cpp configuration found by the compiler, not a weak
   hand-selected baseline.
4. **Static before dynamic.** Static MoE placement remains the primary expert
   optimization until a dynamic candidate beats the fully optimized static
   plan.
5. **Fail closed.** Model, tensor, runtime, hardware, plan, or capability
   mismatches abort rather than silently dropping compiler passes.
6. **Explain every plan.** Accepted and rejected candidates retain mechanisms,
   measurements, constraints, trade-offs, and provenance.
7. **Bounded extensibility.** Typed IRs, model adapters, compiler passes, and
   cost-model interfaces provide customization without introducing a general
   compiler DSL in the first release.

## 3. Compiler boundary

### 3.1 Inputs

- immutable GGUF path, size, and SHA-256;
- model-family adapter;
- pinned llama.cpp runtime and build identity;
- measured NVIDIA GPU and CUDA profile;
- workload objective and benchmark corpus;
- exact or approximate policy;
- context target, concurrency, VRAM reserve, and quality constraints;
- compiler version and pass configuration.

The first workload objective is maximum single-request decode TPS at
concurrency one. Aggregate server throughput becomes a later objective and may
not reuse single-request claims without new measurements.

### 3.2 Outputs

The compiler emits a versioned `ExecutionPlan` containing:

- all input identities and hashes;
- selected weight quantization;
- GPU layer placement;
- static expert-residency configuration;
- optional dynamic-cache configuration;
- KV-cache representation;
- CUDA graph, Flash Attention, batch, thread, and build settings;
- optional MTP or speculative-decoding configuration;
- exact launch arguments and environment;
- projected and measured memory accounting;
- evidence and rejected-candidate provenance;
- validation gates and a compatible fallback plan.

The ExpertFlow llama.cpp fork consumes this declarative plan through a stable
runtime interface. The compiler does not generate or rewrite C++ source for
individual models.

### 3.3 Fork governance

The runtime fork is maintained as a small, reviewable patch stack over an exact
upstream llama.cpp base commit. Each ExpertFlow release records the upstream
base, ordered patch identities, build configuration, exported interface, and
compatible compiler-plan schema.

Upstream synchronization occurs only at explicit integration milestones, never
implicitly during plan compilation or measurement. A candidate upstream update
is evaluated in an isolated branch through this sequence:

1. reproduce the old upstream stock baseline and accepted ExpertFlow plan;
2. build and test the new unmodified upstream revision;
3. rebase or reapply each ExpertFlow patch independently;
4. run feature-off equivalence, native contracts, plan compatibility, exactness,
   performance, memory, and cleanup gates;
5. promote the new base only after the compatibility matrix passes.

New upstream mechanisms such as MTP are first validated on unmodified upstream,
then integrated through the smallest stable hook needed by the plan interface.
Where practical, generally useful hooks are proposed upstream to reduce fork
divergence. Failed upstream updates remain quarantined; the current supported
base continues to receive reproducibility support.

The project tracks patch-stack size, touched subsystems, unresolved upstream
conflicts, and time since the last evaluated upstream base. If the fork can no
longer be rebased without broad scheduler, graph, allocator, or kernel rewrites,
the release stops and requires a new runtime-architecture decision rather than
quietly accumulating divergence.

## 4. Intermediate representations

### 4.1 ModelIR

`ModelIR` normalizes model-family details:

- model identity, family, quantization, tensor provenance, and alignment;
- ordered transformer layers and dense/MoE classification;
- router type, expert count, top-k, and routing invariants;
- expert tensor bundles, shapes, encodings, and exact packed byte layouts;
- attention, KV-cache, recurrent-state, and rollback semantics;
- native MTP or compatible draft capabilities;
- operations and layouts supported by the pinned runtime.

Model adapters may parse family-specific metadata and tensor names. Generic
optimization passes may consume only normalized `ModelIR` fields, never raw
Gemma- or Qwen-specific names.

### 4.2 HardwareIR

`HardwareIR` contains:

- GPU identity, compute capability, and usable VRAM;
- driver, CUDA, and CUDA Runtime API identity;
- supported llama.cpp CUDA features and kernels;
- measured allocation frontier and mandatory reserve;
- measured transfer, synchronization, and selected kernel curves;
- thermal and repeat-variance metadata.

### 4.3 WorkloadIR

`WorkloadIR` contains:

- objective and strongest stock comparison rule;
- prompt, prefill, decode, and context shapes;
- concurrency, initially fixed at one;
- benchmark corpus and immutable workload hash;
- exactness policy and approximate quality budget;
- repetition, warmup, pairing, and stop rules.

### 4.4 CandidatePlan and ExecutionPlan

A `CandidatePlan` may be estimated, unmeasured, invalid, rejected, or measured.
Only a measured candidate that passes all constraints can be sealed as an
`ExecutionPlan`. Estimated values remain labeled estimates and may not be
promoted into measured claims.

## 5. Compiler architecture

The compiler uses a typed pass pipeline:

```text
GGUF + runtime + GPU + workload
                |
                v
       Normalize into typed IRs
                |
                v
   Capability and constraint analysis
                |
                v
     Candidate-generating passes
                |
                v
 Measurement-guided bounded plan search
                |
                v
       Validation and plan sealing
                |
                v
     ExpertFlow llama.cpp execution
```

Each pass declares required IR capabilities, consumed analyses, modified plan
fields, incompatible passes, and validation obligations. The pass manager
topologically orders compatible passes and rejects missing or cyclic
dependencies.

The first release exposes typed configuration for pass selection and bounds.
It does not expose arbitrary code execution or a general optimization DSL.

## 6. Pass order

### 6.1 Inspect and normalize

Parse the GGUF and runtime identities, select the model adapter, build the IRs,
and reject unsupported architecture, quantization, layout, or capability
combinations.

### 6.2 Stock baseline search

Search supported stock llama.cpp build and runtime settings. Measure decode
TPS, prompt TPS, inter-token latency, end-to-end time, VRAM, output identity,
and cleanup. Freeze the strongest valid candidate as the optimization floor.

### 6.3 Memory-plan compilation

Select the model quant, KV format, context allocation, runtime state, and
mandatory reserve before expert placement. Approximate KV formats require an
explicit approximate policy and quality gate.

### 6.4 Static MoE placement

Profile layer costs, rank candidates by expected saved execution time per VRAM
byte, search bounded layer/expert placements, and lower the chosen placement
into the pinned runtime. Reproduce the existing twelve-layer Gemma result as a
regression target before searching for a better plan.

### 6.5 CUDA execution autotuning

Search CUDA graphs, Flash Attention, batch and microbatch size, CPU threads,
tensor offload, supported kernels, pinned staging, and build flags. Build flags
that affect numerical behavior are immutable plan provenance.

### 6.6 MTP and speculative decoding

Run only when the model and runtime declare compatible support. Search bounded
draft methods and windows using acceptance rate, verifier work, extra KV/state
memory, decode TPS, and MoE cold-expert union. Disable speculation when it
reduces TPS or destabilizes memory or residency.

Upstream llama.cpp mechanisms are used before custom MTP implementations.

### 6.7 Two-table dynamic residency

Dynamic expert caching is a conditional final optimization rather than the
default architecture.

The initial exact policy contains:

1. a global prior table keyed by model, quant, workload/domain, layer, and
   expert, learned only from training evidence; and
2. a session table built from current routing, reuse, hit/miss, and phase data.

Physical capacity is divided between pinned prior residents and adaptive
session residents. Rebalancing occurs at request or prefill/decode boundaries
where possible. Genuine misses retain exact blocking loading. The first pass
contains no learned predictor and no asynchronous speculative transfer.

Dynamic residency is retained in a maximum-TPS plan only if it beats the
optimized static+KV+CUDA+MTP plan under the same exact runtime contract. A
memory-saving result may be emitted as a separately named capacity profile but
may not be presented as the maximum-TPS result.

### 6.8 Bounded joint search

After individual passes select provisional winners, perform a bounded local
search around their interacting settings. The compiler must not attempt an
unbounded Cartesian search. Search budgets, candidate counts, stop rules, and
all rejected combinations are recorded.

### 6.9 Validation and sealing

Re-run exactness or quality gates, paired repetitions, memory bounds, cleanup,
runtime identity, and plan replay. Seal the winner and compatible fallback only
after every declared obligation passes.

## 7. Exact and approximate policies

### 7.1 Exact profiles

Exact profiles require:

- compatible immutable identities;
- declared deterministic sampling conditions;
- exact prompt and generated token identity against the appropriate baseline;
- exact router identity where the pass claims router preservation;
- stable allocations and complete cleanup;
- no unapproved tensor repacking or numerical path change.

### 7.2 Approximate profiles

Weight-quant changes, low-bit KV formats, TurboQuant-style compression, or any
other quality-changing pass require:

- explicit `approximate` policy;
- separate plan and performance claims;
- frozen perplexity, task, long-context, and retrieval gates appropriate to the
  model;
- an explicit maximum quality budget;
- no fallback from an exact request into an approximate plan.

TurboQuant remains an experimental later pass. Existing llama.cpp KV formats
must be measured first. A production TurboQuant pass requires compatible GPU
kernels and actual compressed GPU storage; a Python or CPU reference
implementation is insufficient.

This separation is a deliberate product invariant, not a temporary limitation.
Any KV representation that changes stored numerical values, including Q8 or Q4,
is approximate even when a finite evaluation corpus produces identical tokens.
Token parity is a required regression signal but cannot prove numerical
equivalence for unseen contexts. Exact requests retain the baseline KV datatype
and numerical path. Users may create named approximate profiles with explicit
quality budgets, but the compiler never silently softens an exact request.

## 8. Measurement database and cost model

Measurements are keyed by:

```text
model hash
quant profile
runtime and build hash
GPU identity
driver and CUDA identity
compiler version
workload hash
execution-plan hash
```

The database stores raw samples, summaries, environment state, timestamps,
commands, exit codes, hashes, exactness/quality results, and artifact paths.

The cost model proposes and prunes candidates; it does not declare winners.
Its estimates include VRAM, transfer bytes, execution time, expected hits,
speculative verifier cost, and constraint risk. Only measured finalists may be
selected.

### 8.1 Calibration and feedback loop

The cost model begins with analytical byte accounting and conservative priors
from compatible measurements. It is calibrated separately for each complete
measurement key; observations from a different model, quant, runtime build,
GPU, CUDA identity, or workload may initialize a prior but may not be treated
as calibrated evidence.

After each measured candidate, the compiler records prediction residuals for
TPS, latency, memory, transfer cost, and pass-specific counters. It updates the
local calibrated model, uncertainty bounds, and candidate ranking before the
next measurement batch. Model versions and calibration inputs are immutable
plan provenance so a search can be replayed.

Finite measurement budgets include an exploration quota. The compiler must
measure:

- the predicted winner;
- boundary candidates near hard constraints;
- at least one diverse or uncertainty-maximizing candidate per active pass;
- periodic sentinel candidates that the cost model ranked below the finalist.

Unknown or high-uncertainty regions may be deprioritized but not eliminated by
an unsupported point estimate. If sentinel residuals exceed a declared error
threshold, pruning is suspended, the affected search region is widened, and
the compiler either recalibrates or emits an inconclusive result. Small search
spaces use exhaustive measurement instead of a cost model.

## 9. Model-family adapters

Adapter order:

1. existing Gemma 4 26B A4B quant and layout;
2. another Gemma quantization;
3. a Qwen MoE architecture with conventional attention;
4. a Qwen hybrid or recurrent architecture after state, rollback, and MTP
   semantics have a dedicated specification.

An adapter supplies model facts and lowering hooks. It must not implement its
own placement, cache, KV, or speculative optimizer.

Multi-family support is accepted only when the same generic passes compile and
validate at least two model families without family checks leaking into pass
logic.

## 10. Error handling and fallback

Compilation stops with structured diagnostics when:

- an identity, tensor layout, architecture, or capability is unsupported;
- measured memory exceeds the plan or reserve;
- an exactness or quality gate fails;
- a benchmark process fails or leaves residual state;
- a cost-model estimate cannot be reconciled with measured accounting;
- the requested optimization objective has no valid candidate.

Runtime plan validation occurs before model execution. A sealed plan may name a
compatible measured stock fallback. Fallback is allowed only when its model,
runtime, hardware, workload, and policy identities match; otherwise execution
stops and requests recompilation.

## 11. Verification strategy

### 11.1 Unit and contract tests

- GGUF and adapter parsing;
- checked tensor-byte arithmetic and alignment;
- IR serialization and migrations;
- pass dependencies and incompatibilities;
- constraint propagation and rejection;
- plan identity and hash validation;
- deterministic candidate ordering;
- dynamic-cache policy replay;
- cost-model accounting.

### 11.2 Native runtime tests

- assertion-active planner and mapping tests;
- exact tensor bindings and bounds;
- feature-off equivalence;
- allocation, teardown, and failure cleanup;
- CUDA capability and plan rejection;
- MTP state and rollback where supported.

### 11.3 End-to-end gates

- strongest-stock reproduction;
- existing twelve-layer Gemma regression target;
- exact paired output comparison;
- repeated decode-TPS and latency measurements;
- process-owned memory and reserve;
- frozen quality evaluation for approximate plans;
- clean-checkout plan replay.

## 12. Development phases

Phases 0 through 3 are the required compiler spine and the only sequential
critical path. After Phase 3, the compiler is a usable static-placement
product. Phases 4 through 8 are independent optimization tracks with declared
prerequisites, budgets, and stop rules; a no-go or stalled track emits an
evidence-backed rejection and does not block unrelated later tracks. For
example, CUDA autotuning may reject every candidate without preventing KV
experiments or a new model adapter.

The Phase 9 product commands and plan cache begin incrementally after Phase 3.
Joint cross-pass search is added only for passes that independently produced a
valid winner. This prevents one research-grade optimization from delaying the
entire compiler.

### Phase 0: clean compiler baseline

Consolidate provenance, pin the ExpertFlow llama.cpp fork and CUDA build, repair
the Python environment, separate historical evidence from compiler artifacts,
and provide one command that reproduces the strongest stock and current
ExpertFlow configurations.

Exit gate: clean-checkout reproduction with declared output, TPS, and VRAM
ranges.

### Phase 1: compiler core

Implement the IRs, Gemma adapter, pass manager, capability/constraint system,
plan schemas, versioning, validation, and diagnostics.

Exit gate: `expertflow compile --model ... --objective decode-tps` emits and
validates a behavior-preserving baseline plan.

### Phase 2: measurement and cost-model engine

Implement the local evidence database, strongest-stock search, hardware
profiling, candidate accounting, and repeat protocol.

Exit gate: automatically select and reproduce the strongest measured stock
configuration within a declared tolerance.

### Phase 3: static MoE placement compiler

Convert existing layer diagnostics and static placement into generic analyses,
candidate generation, lowering, and validation.

Exit gate: match or exceed the authoritative existing Gemma 28.13 decode-TPS
result under its compatible protocol.

### Phase 4: CUDA execution autotuning

Add bounded search for graphs, Flash Attention, batches, threads, kernels,
offload, staging, and build flags.

Exit gate: retain only statistically supported exact TPS improvements with
stable memory.

### Phase 5: KV and shared memory-budget compiler

Measure supported llama.cpp KV formats, context growth, quality, and placement
interactions. Re-run static placement with released memory. Add TurboQuant only
as a later approximate experiment with GPU storage and kernels.

Exit gate: select a better valid TPS/memory plan or emit an evidence-backed
rejection.

### Phase 6: additional model adapters and quants

Add another Gemma quant and then Qwen MoE through normalized adapters.

Exit gate: generic passes compile and validate two model families and multiple
quants without model-specific pass branches.

### Phase 7: MTP and speculation

Detect upstream llama.cpp capabilities, search bounded speculative settings,
and measure acceptance, verifier work, memory, expert union, and TPS.

Exit gate: at least 10% decode-TPS improvement, stable memory, and no exactness
regression against the same plan without speculation.

### Phase 8: two-table dynamic expert residency

Implement the global/session policy without prediction, compile its memory
split, and compare it against the final optimized static plan.

Exit gate: measured maximum-TPS improvement. Memory-only success is emitted as
a separate capacity profile.

### Phase 9: joint optimizer and product surface

Expose `inspect`, `profile`, `compile`, `benchmark`, `validate`, `explain`,
`run`, and `serve`; add bounded joint search, plan caching, decision reports,
and safe fallback.

Exit gate: a user supplies a supported GGUF and receives a reproducible,
validated single-GPU maximum-TPS plan without manually selecting layers or
runtime flags.

## 13. Deferred work

- CPU or Vulkan performance compilation;
- multi-GPU execution;
- non-llama.cpp backends;
- AirLLM-style weight streaming;
- arbitrary compiler DSL or untrusted plugins;
- production multi-user objective optimization;
- custom MTP before upstream mechanisms are exhausted;
- predictive or asynchronous expert prefetch without a new timing mechanism;
- approximate execution without an explicit quality policy.

## 14. Upstream mechanism references

- AirLLM is excluded, but its layer-wise loading distinction informed the
  capacity-versus-throughput boundary:
  <https://github.com/lyogavin/airllm/blob/main/README.md>
- llama.cpp provides the pinned runtime foundation and upstream speculative
  mechanisms:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- TurboQuant remains an experimental KV research direction rather than an
  immediate production dependency:
  <https://github.com/scos-lab/turboquant>

## 15. Final success definition

ExpertFlow succeeds as an inference compiler when it can take a supported GGUF,
one NVIDIA CUDA GPU, and a declared workload; explain and measure candidate
optimizations; emit a sealed compatible plan; and reproduce the strongest
valid single-request decode-TPS result without manual layer or flag selection.

The compiler is not required to enable every pass. An evidence-backed rejection
of dynamic caching, KV compression, or MTP for a particular model and machine is
a correct compiler result.
