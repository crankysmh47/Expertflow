# Product Spec: ExpertFlow Local — Predictive Memory Hierarchy for Sparse MoE Models and Local Agents

**Document type:** Product + technical design spec
**Status:** Draft v0.11 — layered delivery, dual-demo contingency, and video-first deadline locked
**Date:** 2026-07-14
**Working name:** ExpertFlow Local
**Primary proof-of-concept target:** Gemma 4 26B A4B, text-only, batch size 1
**Post-hackathon research target:** Qwen3.5-35B-A3B
**Initial hardware target:** Single 16 GB VRAM GPU, sufficient CPU RAM, CPU→GPU transfer over PCIe
**Core principle:** Prediction affects loading order only. The true model router remains authoritative.
**Hackathon track:** Developer tools
**Critical-path runtime:** Hugging Face/PyTorch host runtime plus a thin custom ExpertFlow expert-cache backend; no new full inference engine

---

## 0. Revision Summary

### 0.1 v0.11 layered-delivery correction

- Reduced the hackathon commitment from one large all-or-nothing product to four independently demoable delivery layers.
- Locked **Layer 1 Observatory** and **Layer 2 Recommendation + Replay** as the submission floor. The live runtime remains the competitive target, not a prerequisite for a coherent submission.
- Added a complete three-minute **Observatory-only contingency demo**, equal in status and preparation quality to the live-runtime demo.
- Reclassified the agent-runtime and VRAM-headroom narrative as reporting derived from existing memory accounting. It does not require a KV-cache experiment, a session scheduler, or a new runtime subsystem.
- Kept the public product loop but implemented it through one shared pipeline. `profile`, `simulate`, and `recommend` may be separate CLI views over the same artifacts rather than separate products.
- Made the causal replay report the minimum UX. A richer synchronized live dashboard is a target only after the evidence and runtime paths are stable.
- Added hard cutoffs: the live-runtime path freezes by 2026-07-18; all product work freezes by 2026-07-19; 2026-07-20 is reserved for recording, editing, uploading, and verifying the video.
- Added explicit stop rules that force the project onto the Observatory-only submission when the live path threatens the evidence, product experience, or video schedule.

### 0.2 Prior decisions retained

The following decisions remain locked from v0.6–v0.10:

```
Gemma 4 26B A4B, text-only, batch size 1
4-bit-first reference profile
Hugging Face/PyTorch host runtime
custom work limited to routed-expert storage, movement, policy, and telemetry
true router remains authoritative
Layer 1 locality and PCIe evidence before runtime engineering
same-runtime reactive and LRU baselines
measured-versus-estimated separation
judge-runnable tiny-model smoke test and bundled trace replay
MTP/lookahead is stretch only
Qwen, llama.cpp, multimodal, and learned online policy are post-hackathon
```

---

## 1. Product Summary

ExpertFlow Local is a developer tool and runtime layer that turns sparse model weights into a **managed memory hierarchy** instead of treating insufficient GPU VRAM as a hard deployment boundary.

Sparse Mixture-of-Experts models may contain far more total parameters than they activate for any one token. On a local agent machine, those routed weights compete with KV/model state, transfer buffers, and other active sessions for the same scarce VRAM. ExpertFlow measures the model's real routed working set, models the target machine's memory and PCIe behavior, recommends a cache policy, and—when the live backend is enabled—moves likely experts into VRAM before the real router needs them.

The product therefore answers two linked questions:

```
Which experts must be ready to avoid stalls?
How much VRAM can remain available for active agent state and additional sessions?
```

The default mode preserves exact model behavior:

```
The true MoE router decides which experts are used.
ExpertFlow decides only which experts should be loaded early.
A wrong prediction causes latency, not an incorrect answer.
```

### 1.1 One-sentence positioning

> **ExpertFlow manages sparse model weights as a predictive memory hierarchy, keeping the right experts ready while preserving scarce VRAM headroom for active agent state, longer working contexts, or additional local sessions.**

Do not describe the project as "faster disk loading." Active decode uses GPU VRAM, bounded pinned CPU memory, and resident CPU memory; SSD is reserved for initial loading or emergency backing only.

### 1.2 Layered hackathon product

The hackathon build is intentionally layered. Every completed layer produces a runnable artifact and a complete user-facing result.

#### Layer 0 — Empirical gate, internal but immediate

```
load the pinned 4-bit profile
validate router tracing
measure expert sizes and PCIe behavior
run the stratified locality dry run
produce PROCEED / CONDITIONAL / OBSERVATORY-ONLY decision
```

Layer 0 exists to prevent the team from spending the week on an unsupported runtime hypothesis.

#### Layer 1 — ExpertFlow Observatory, required

```
expertflow doctor
router trace collection or bundled replay
locality, reuse-distance, and working-set analysis
reactive/static/LRU/session/oracle simulation
Capability / Performance-potential / Mechanism report
```

This layer answers: *Is predictive expert caching viable for this model on this machine?*

#### Layer 2 — Recommendation + causal replay, required

```
machine-specific expert-cache recommendation
measured VRAM allocation and remaining-headroom report
one causal expert timeline
bundled replay that runs without the full model
static or lightweight interactive HTML report
```

This layer turns the research result into a developer product. It does not require a live expert-cache runtime.

#### Layer 3 — Exact live runtime, competitive target

```
blocking reactive baseline
preallocated GPU expert slots
session-aware/LRU residency
asynchronous prefetch
exact fallback and movement telemetry
same-runtime measured comparison
```

Layer 3 is built only while Layer 1 and Layer 2 remain complete and video-ready.

#### Layer 4 — Adaptive lookahead/MTP, stretch

Layer 4 begins only when Layer 3 is exact, faster, reproducible, and frozen early enough to protect the video schedule.

### 1.3 Minimum product loop and implementation economy

The public journey remains:

> **Inspect hardware → profile model → compare policies → receive recommendation → replay or run → verify**

The CLI may expose this as separate commands:

```bash
expertflow doctor
expertflow profile --model google/gemma-4-26b-a4b-it
expertflow simulate --trace runs/profile/trace.parquet
expertflow recommend --simulation runs/profile/simulation.json
expertflow replay --run runs/profile
expertflow run --config runs/profile/recommended.yaml      # Layer 3 only
expertflow verify --baseline runs/reactive --candidate runs/expertflow  # Layer 3 only
```

These commands must share one artifact pipeline and configuration schema. The team must not build separate analysis engines, storage formats, or frontends for each command.

The minimum shippable UX is a generated report containing:

```
hardware readiness
locality verdict
policy comparison
recommended cache allocation
remaining measured VRAM headroom
one causal ready/late/blocking timeline
reproduction command and manifest
```

A richer live dashboard is permitted only after this report is complete.


### 1.4 Claim hierarchy

The submission uses three explicit claim levels.

#### Proven in the hackathon

> **ExpertFlow profiles and simulates sparse expert movement on a 16 GB GPU target, and—only if Layer 3 measurements pass—uses an exact predictive expert cache to reduce blocking transfer time versus a same-runtime reactive baseline.**

Do not claim a runtime improvement until it is measured.

#### Immediate product value

> **ExpertFlow helps developers determine whether a sparse model is viable on their hardware, compare cache strategies, receive a machine-specific VRAM allocation, and inspect every expert transfer that caused or avoided a stall—while showing how much memory remains available for active agent state or additional sessions.**

This is valuable even when a particular model has weak locality and the correct recommendation is not to enable predictive caching.

#### Long-term implication

> **A predictive memory hierarchy could lower the minimum hardware tier required to deploy future large-total-parameter, low-active-parameter models and make stronger private agents practical on individual workstations by sharing VRAM deliberately between routed weights and active agent state.**

This is a future implication, not a hackathon benchmark claim. Do not imply that the Gemma result proves a specific hundred-billion-parameter model will run efficiently on a particular workstation GPU.

### 1.5 Audience and impact

Primary users:

```
local-agent and local-AI application developers
agent-runtime and ML-systems researchers
small AI companies and independent builders
privacy-sensitive organizations
teams with sufficient CPU RAM but insufficient GPU VRAM for a target sparse model
```

The practical problem is not merely inference cost. It is that a model may have a manageable active computation but an impractical total VRAM footprint. ExpertFlow tests whether the routed working set and transfer deadlines make that model usable on the available machine.

Potential downstream uses include private coding and research agents, longer uninterrupted tool-using sessions, several local agent sessions sharing one workstation, offline experimentation, regulated or disconnected environments, local model evaluation, and applications that would otherwise require multi-GPU servers or cloud inference. ExpertFlow does not claim that a measured number of free megabytes automatically equals a particular context length; it reports the headroom and leaves architecture-specific conversion to measured runtime profiles.

### 1.6 Delivery outcomes

The project has two prepared submission outcomes.

```
Outcome A — Observatory product, guaranteed:
  Layer 1 + Layer 2 complete
  real Gemma traces and hardware measurements
  policy simulator and recommendation
  causal replay report
  explicit verdict on whether live predictive caching is worthwhile

Outcome B — Runtime product, target:
  Outcome A plus Layer 3
  exact live expert cache
  measured reactive/LRU/ExpertFlow comparison
  runtime movement timeline

Outcome C — Lookahead extension, optional:
  Outcome B plus Layer 4 only when it produces a measured incremental gain
```

Outcome A is not described as a failed runtime. It is a model-and-hardware profiler that prevents developers from committing to an offloading architecture before measuring locality, transfer deadlines, and VRAM trade-offs.

### 1.7 Three-minute demo A — exact live runtime succeeds

```
0:00–0:20  Constraint
  A sparse model's routed weights and active agent state compete for 16 GB of VRAM.
  Show expertflow doctor and the measured memory envelope.

0:20–0:45  Observatory
  Show the real router trace, layer-specific working set, and oracle headroom.

0:45–1:15  Reactive baseline
  Expert is requested while absent; PCIe copy starts after demand; GPU stalls.

1:15–1:50  ExpertFlow runtime
  Expert is predicted, transferred during useful compute, then selected by the real router.
  Show READY HIT and exact output parity.

1:50–2:20  Evidence
  Compare reactive, LRU, and ExpertFlow using TPS, p95 latency, blocking transfer time,
  ready-hit rate, peak VRAM, and remaining headroom.

2:20–2:45  Product
  Show the generated machine-specific recommendation and reproduction command.

2:45–3:00  Implication
  Predictive sparse-weight management lowers the hardware boundary while leaving
  measured headroom that a compatible agent runtime may allocate to active state.
```

### 1.8 Three-minute demo B — Observatory-only contingency

This script must be rehearsed and asset-complete by 2026-07-18, even while Layer 3 is still being attempted.

```
0:00–0:20  The expensive systems mistake
  Developers see a sparse model that does not fit in VRAM and immediately begin
  implementing offload logic without knowing whether its routing is cacheable.

0:20–0:45  One-command hardware and model inspection
  Run expertflow doctor and profile. Show the exact 4-bit memory envelope,
  transfer bandwidth, and routed-expert trace.

0:45–1:20  The locality result
  Show concentration, reuse distance, working-set size, and topic-shift behavior.
  Explain whether locality is strong, weak, or layer-dependent.

1:20–1:55  Policy laboratory
  Replay the same trace under reactive, static, LRU, session-aware, and oracle policies.
  Show ready, late, and blocking events on the causal timeline.

1:55–2:25  Machine-specific recommendation
  Show the recommended expert slots, pinned-memory budget, expected cold MB/token,
  predicted stalls, uncertainty, and remaining measured VRAM headroom.

2:25–2:45  Honest decision
  State the evidence-backed verdict:
    PROCEED — live predictive caching has measurable headroom;
    CONDITIONAL — only selected layers/domains justify it; or
    DO NOT ENABLE — locality/PCIe economics do not support it on this profile.

2:45–3:00  Product value
  ExpertFlow turns weeks of speculative runtime work into a reproducible hardware-aware
  decision, and its replay bundle lets any judge inspect the mechanism without our GPU.
```

If Layer 3 is incomplete or does not beat the baselines by the cutoff, Demo B becomes the official submission script immediately. Simulated metrics remain visibly labeled as estimates.

### 1.9 Submission clock and hard freeze policy

```
2026-07-14: Layer 0 — environment, INT4 load, router hook, locality dry run
2026-07-15: Layer 1 evidence bundle, simulator, final runtime go/no-go decision
2026-07-16: freeze Layer 1; complete minimum replay report and recommendation schema
2026-07-17: freeze Layer 2; rehearse and screen-test Demo B; continue Layer 3 if justified
2026-07-18: Layer 3 cutoff; keep only exact, measurable runtime work that already functions
2026-07-19: all product, benchmark, README, dashboard, and demo assets freeze; rehearse both scripts
2026-07-20: VIDEO-ONLY DAY — record, edit, upload, process, and verify the final video
2026-07-21: reproduction check, submission forms, link verification, and emergency corrections only
2026-07-22 05:00 PKT: external deadline assumption; do not plan active work until this point
```

No feature implementation is scheduled for video day. A broken result discovered on July 20 is removed from the story rather than repaired through new architecture work.


---

## 2. Strategic Model Decision

### 2.1 Primary proof-of-concept: Gemma 4 26B A4B

Gemma 4 26B A4B is the correct first target because it is smaller and simpler than Qwen3.5-35B-A3B while still being an MoE model large enough to stress 16 GB VRAM inference.

Officially documented Gemma 4 26B A4B properties:

| Property | Value |
|---|---:|
| Total parameters | 25.2B |
| Active parameters | 3.8B |
| Layers | 30 |
| Sliding window | 1024 tokens |
| Context length | 256K tokens |
| Vocabulary size | 262K |
| Expert count | 8 active / 128 total + 1 shared |
| Supported modalities | Text, image |
| Vision encoder parameters | ~550M |

Source: Google AI for Developers Gemma 4 model card, accessed 2026-07-09:
https://ai.google.dev/gemma/docs/core/model_card_4

### 2.1.1 MVP modality scope

Gemma 4 26B A4B is natively multimodal. The MVP is text-only.

MVP decision:

```
Vision encoder: excluded from MVP runtime scope
Image inputs: unsupported in Hackathon Stages 1–4
Vision-text global priors: schema-reserved but disabled in MVP
VRAM accounting: include an optional row for the vision encoder but do not reserve VRAM for it
```

Rationale:

```
The first research question is whether predictive routed-expert caching helps text MoE inference on 16 GB VRAM.
Adding image processing and vision encoder residency would confound the memory budget.
```

Later multimodal phase options:

```
Option A: keep vision encoder CPU-side and transfer image embeddings to the decoder
Option B: temporarily load vision encoder for image prefill, then evict before text decode
Option C: keep vision encoder resident only on GPUs with enough spare VRAM
```

### 2.2 Later research target: Qwen3.5-35B-A3B

Qwen3.5-35B-A3B remains attractive but should not be the first implementation target. Its hybrid Gated DeltaNet + sparse MoE architecture makes memory accounting and runtime integration materially harder.

Officially documented Qwen3.5-35B-A3B properties:

| Property | Value |
|---|---:|
| Total parameters | 35B |
| Activated parameters | 3B |
| Layers | 40 |
| Hidden dimension | 2048 |
| Expert count | 256 |
| Activated experts | 8 routed + 1 shared |
| Expert intermediate dimension | 512 |
| Hidden layout | 10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)) |
| MTP | Trained with multi-steps |
| Native context length | 262,144 tokens |
| Extended context length | up to 1,010,000 tokens |

Source: Qwen3.5-35B-A3B Hugging Face model card, accessed 2026-07-09:
https://huggingface.co/Qwen/Qwen3.5-35B-A3B

### 2.3 Product decision

```
Hackathon Stages 1–4: Gemma 4 26B A4B only.
Post-hackathon Stage 5+: Qwen3.5-35B-A3B only if Gemma results are compelling.
```

---

## 3. Problem

Large sparse MoE models create a mismatch between **total parameter storage** and **active computation**. A machine may have enough GPU compute and VRAM for the backbone, runtime state, and a useful expert working set, while still lacking enough VRAM to hold every routed expert simultaneously.

Existing local-inference systems use combinations of quantization, CPU offload, memory mapping, static placement, and reactive movement. The defensible problem statement is not that all existing systems stream weights from disk. It is that a reactive miss path discovers missing weights too late:

```
router selects expert
expert is not ready in VRAM
runtime schedules or performs a host-to-device transfer
GPU execution waits for the required expert
computation resumes
```

Every late expert becomes a latency event. Static hot/cold placement also leaves performance on the table when the working set changes by layer, domain, conversation, or topic phase.

ExpertFlow asks a concrete systems question:

> **Can the routed expert working set be predicted and scheduled early enough that host memory becomes a managed extension of VRAM rather than a source of repeated blocking stalls?**

The tool must also be willing to answer "no" for a model/hardware profile when locality is weak, the working set is too large, or PCIe deadlines cannot be met.

---

## 4. Core Insight

Sparse model weights should be managed as a hierarchy:

```
GPU VRAM:              experts needed now or likely before a near deadline
Pinned CPU memory:     warm experts prepared for fast DMA
Resident CPU memory:   complete cold expert store
SSD:                   initial loading or emergency backing, not normal decode
```

For a stable conversation, expert routing may exhibit exploitable locality across tokens, layers, domains, and topic phases. Experts do not need human-readable identities such as "the Python expert" for this locality to be useful.

```
Do not permanently disable experts.
Do keep likely experts hot.
Do let unused experts become cold.
Do allow the true router to recover any cold expert if needed.
```

The correct mental model is:

> **MoE expert offloading is predictive, read-only, layer-partitioned, deadline-aware working-set management inside a shared VRAM budget.**

The routed-expert cache is not the only consumer of that budget. A local agent runtime must also reserve space for KV/model state, scratch buffers, transfer staging, fragmentation margin, and possibly other sessions. ExpertFlow optimizes expert readiness without pretending that maximizing expert-cache size is always the best use of the last gigabyte of VRAM.

### 4.1 Global prior plus session adaptation

The two-table design supports both cold start and personalization:

```
GlobalPriorTable
  gives the machine a reasonable initial working set before session evidence exists

SessionExpertTable
  adapts quickly to the current conversation and topic phase
```

The global prior must never overpower current-session evidence. The session table must never alter the real router's choice. Both affect residency priority only.

### 4.2 Product and impact boundary

The hackathon proves the mechanism on Gemma 4 and a 16 GB GPU target. The broader thesis is that the same architecture may lower the hardware boundary for future sparse models whose active subset is much smaller than their total parameter set.

Required wording discipline:

```
Allowed now:
  "ExpertFlow measures whether this model has a cacheable routed working set."
  "ExpertFlow reduced measured blocking transfer time on this tested profile."  # only after proof
  "The architecture may generalize to much larger sparse models."
  "The tested policy leaves X measured MB of VRAM headroom after required runtime and cache allocations."

Not allowed without evidence:
  "All current offloaders stream from disk."
  "ExpertFlow makes any MoE fit on one GPU."
  "A Gemma result proves a specific frontier model will run interactively on one workstation."
  "X MB of measured headroom automatically provides Y more context tokens or Z concurrent agents."
```

---

## 4A. Expert-Routing Data Source Decision

### 4A.1 Direct answer

Conversation-to-expert routing traces are not ordinary text datasets. The selected expert IDs are specific to:

```
model family
checkpoint/revision
layer numbering
router implementation
tokenizer and chat template
precision/quantization path
runtime implementation
sampling settings
conversation history
```

A routing trace from DeepSeek, Qwen, Llama, or a different Gemma checkpoint cannot be treated as a label set for Gemma 4 26B A4B. Expert ID 17 in one model has no defined relationship to expert ID 17 in another model.

Product decision:

```
Authoritative Gemma training data:
  collect directly from the exact Gemma checkpoint and runtime configuration.

Public traces:
  use for schema design, tooling validation, simulator tests,
  baseline reproduction, and research comparison only.
```

As of 2026-07-10, the reviewed public sources did not provide a suitable conversation-level routing-trace dataset for `google/gemma-4-26B-A4B-it`.

### 4A.2 Public sources we can use

| Source | What it provides | Models | Use in ExpertFlow | Limitation |
|---|---|---|---|---|
| MoE-Beyond repository and paper | Collector scripts, prompt-level CSV traces, cache simulator, predictor training code and pretrained artifacts | DeepSeek-V2-Lite | Reproduce trace schema; inspect training pipeline; validate simulator assumptions | Expert IDs and dynamics are DeepSeek-specific; repository licensing must be checked before redistribution |
| MoE-Infinity repository and paper | Open-source activation-aware tracing, caching, prefetching, offloading runtime | DeepSeek, Mixtral, Qwen3-MoE, DBRX, Jamba, OLMoE, NLLB-MoE and others listed by the project | Runtime architecture reference; caching and tracing design comparison | Gemma 4 is not listed as a supported model in the reviewed repository version |
| `core12345/MoE_expert_selection_trace` | Per-request, per-token, per-layer selected-expert IDs across benchmarks | DeepSeek-R1, Kimi-K2-Thinking, Llama 4 Maverick, Qwen3-235B | Test analysis code, storage schemas, visualizations, cache simulators | Gated access; no Gemma 4; very different model sizes and serving conditions |
| `ngavhane/moe-beyond/training_data_eamc` | Public example/training CSV files | DeepSeek-V2-Lite | Quick smoke tests for parsing and multi-label training code | Not transferable as Gemma labels |

### 4A.3 Public prompt sources for generating Gemma traces

The input conversations can come from public dialogue datasets, but the expert labels must be generated by running Gemma.

Candidate prompt sources:

```
LDJnr/Pure-Dove
OpenAssistant/oasst1 or oasst2
HuggingFaceH4/ultrachat_200k
allenai/WildChat or a non-toxic/filtered release
LDJnr/Capybara
curated code, math, translation, tool-use, and topic-shift suites
```

Dataset licenses, terms, safety filters, and redistribution rights must be reviewed individually. Prompt text and derived traces should not be redistributed merely because the trace collector is open source.

### 4A.4 Why self-collection is the correct solution

Self-collection gives:

```
exact target-model expert IDs
exact target layer structure
real prefill/decode distinction
real chat-template behavior
real topic-shift behavior
matching sampling configuration
matching router probabilities and selected weights
matching runtime/checkpoint version
```

The collector is a read-only observer. It must not replace, patch, or alter the true router.

---

## 5. Non-Goals

The hackathon MVP must not attempt to:

1. Retrain or fine-tune the base MoE model.
2. Modify the true router's selected experts or weights.
3. Permanently prune experts based on domain guesses.
4. Use SSD as the normal active expert source.
5. Support image or multimodal inputs.
6. Build a new tokenizer, attention implementation, KV/state manager, generation engine, or serving stack.
7. Make a deep `llama.cpp`/GGML fork the critical path.
8. Support multiple model families in the live runtime.
9. Implement Qwen3.5/DeltaNet runtime support during the hackathon.
10. Depend on speculative decoding or MTP for Layer 3 correctness or speedup.
11. Verify speculative tokens using only currently resident experts.
12. Train a heavy neural predictor before heuristic and simulator baselines are complete.
13. Collect a massive trace corpus before a 100–200 conversation pilot establishes locality.
14. Build production multi-user scheduling, distributed serving, continuous batching, or multi-GPU support.
15. Guarantee all-GPU-resident throughput.

Approximate modes may be researched later. The hackathon default is exact.

---

## 6. Hardware Assumptions

### 6.1 Minimum target

```
GPU: 16 GB VRAM
CPU RAM: preferably 64 GB+
Storage: SSD for initial load only, not active expert fetches
PCIe: ideally PCIe 4.0 x16 or better
Batch size: 1 for MVP
Context: short-to-medium for MVP; long context only after profiling
```

### 6.2 Memory hierarchy

```
L0: current GPU compute state
L1: hot routed experts in GPU VRAM
L2: warm experts in pinned CPU RAM
L3: cold experts in normal CPU RAM / resident mmap
L4: SSD only as emergency backing store
```

All quantized routed experts should be in CPU RAM or memory-mapped with pages resident. SSD-backed active expert loading is expected to be too slow for interactive decode.

---

## 7. High-Level Architecture

ExpertFlow has a product/control plane and a runtime/data plane.

### 7.1 Product and control plane

```
expertflow doctor
      ↓
HardwareProfile
  GPU/VRAM, host RAM, PCIe bandwidth, pinned-memory budget, backend compatibility, current VRAM allocation
      ↓
expertflow profile
      ↓
RouterTrace + LocalityReport
  concentration, reuse distance, per-layer working set, deadline feasibility
      ↓
expertflow simulate
      ↓
PolicyComparison
  reactive, static, LRU, session, predictor, oracle
      ↓
expertflow recommend
      ↓
MachineSpecificConfig
  slot counts, victim cache, warm-store budget, reserved runtime/KV-state headroom, admission policy, lookahead setting
      ↓
expertflow run + expertflow verify
      ↓
MeasuredResult + ExactnessReport + ReplayableTimeline
```

A profile run that concludes predictive caching is not worthwhile is a valid product result.

### 7.2 Runtime and data plane

```
Application
  chat/completion API
      │
      ▼
Inference Runtime
  tokenizer
  prefill loop
  decode loop
  optional speculative verifier
      │
      ▼
ExpertFlow Cache Layer
  RouterObserver
  TraceCollector
  GlobalPriorTable
  SessionExpertTable
  DomainClassifier
  PredictorRegistry
  PrefetchPlanner
  ExpertCacheManager
  MTPOrDraftWindowController
  PCIeTransferScheduler
  MetricsCollector
      │
      ▼
GPU VRAM                         CPU RAM
  backbone blocks                  quantized routed experts
  routers                          warm pinned staging buffers
  shared experts                   global prior DB
  KV/state                         predictor weights
  hot routed experts               trace logs
  victim cache
  prefetch buffer
```

### 7.3 Product contract

Every stage emits versioned artifacts that the next stage consumes:

```
doctor.json
trace_manifest.json + router_events.parquet
locality_report.json
simulation.json
recommended.yaml
run_manifest.json + expert_movements.jsonl
verification.json
```

The dashboard is a renderer for these artifacts, not a separate source of truth.

---

## 7A. Hackathon Runtime Decision

### 7A.1 Decision

Use a **thin custom ExpertFlow backend inside a pinned Hugging Face/PyTorch execution path**.

```
Keep from the host runtime:
  tokenizer and chat template
  embeddings
  attention/backbone blocks
  KV cache or model state
  Gemma router computation
  sampling and generation loop
  correctness reference path

Implement in ExpertFlow:
  Gemma routed-expert adapter
  CPU expert store
  bounded pinned-memory warm store
  preallocated per-layer GPU expert-slot arena
  cache admission/replacement policy
  asynchronous CUDA copy scheduler
  exact blocking fallback
  event-level metrics recorder
  trace/simulator/dashboard tooling
```

This is neither a full new inference engine nor a wrapper around an unchanged runtime. It replaces the one subsystem the project is evaluating: routed-expert residency and movement.

### 7A.2 Why `llama.cpp` is not the critical path

`llama.cpp` is useful as a later backend and external reference because it provides quantization and CPU+GPU hybrid inference. It is not the first implementation target because dynamic per-expert movement would require invasive changes across GGUF tensor ownership, GGML graph construction, allocators, backend scheduling, CUDA copies, and model-specific MoE execution. That work is too broad for the hackathon and would make router observability harder.

Hackathon rule:

```
Do not block Layer 1 or Layer 3 on a llama.cpp model conversion or upstream patch.
Do not compare ExpertFlow and llama.cpp TPS as the primary scientific result.
Use same-runtime baselines for causal evaluation.
```

A post-hackathon `ExpertBackend` interface may target llama.cpp/GGML.

### 7A.3 Existing-system reuse

MoE-Infinity and KTransformers are architectural references for heterogeneous inference, kernels, and offloading. The hackathon code may reuse permissively licensed components after license review, but must not assume either project already supports the exact Gemma 4 adapter and runtime profile required here.

### 7A.4 Layer 3 execution path

For each routed MoE layer:

```
1. Planner issues prefetch candidates for future layers/tokens using only available information.
2. Transfer scheduler copies selected expert objects into preallocated GPU slots on a copy stream.
3. The true Gemma router executes normally.
4. Expert adapter calls ensure_resident(layer_id, selected_expert_ids, deadline).
5. Ready experts execute immediately.
6. Missing experts trigger an exact blocking load and a recorded stall.
7. Session/cache statistics update after the actual router result.
```

The Layer 3 runtime may initially use PyTorch operations for expert MLP execution. A fused CUDA/Triton kernel is a stretch only after end-to-end correctness and telemetry work.

### 7A.5 Quantization/profile rule — 4-bit first

The first supported deployment profile is **4-bit weight-only routed experts**. This reduces host-memory footprint, PCIe bytes per miss, and the size of the reproducible 16 GB GPU target.

The project is 4-bit first, not 4-bit only:

```
Layer 1 reference profile: one pinned 4-bit format, group size, scale layout, and kernel/backend
Layer 3 first live path: the same 4-bit profile used by reactive, LRU, and ExpertFlow modes
Layer 3 extension: 8-bit weight-only only after 4-bit exactness and telemetry gates pass
Post-MVP: BF16/FP16 diagnostic path when host RAM permits
```

Important engineering constraint:

```
4-bit is easier for capacity and transfer volume.
4-bit can be harder for packed-weight execution and slot replacement.
```

Therefore, the hackathon must support exactly one declared 4-bit representation rather than several interchangeable quantizers. The adapter owns the packed expert object, quantization metadata, GPU-slot layout, and execution call. It must not quantize or repack an expert on every cache miss.

Initial backend candidate:

```
TorchAO stable INT4 weight-only
fixed group size and packing format
source-visible Python/PyTorch integration
no private precompiled ExpertFlow binary
```

This is a candidate, not an assumption. Before Layer 3, implement a focused slot-swap spike that proves:

```
a packed expert object can be serialized in CPU RAM
its bytes and quantization metadata can be copied into a preallocated GPU slot
the public INT4 execution path can consume that slot
repeated replacement does not trigger full-model recompilation or per-miss repacking
selected-expert and output parity remain within the declared quantization tolerance
```

If this spike fails, do not hide the failure by silently changing to 8-bit. Ship the Observatory, publish the failed 4-bit feasibility result, and attempt a different backend only when time remains.

The Layer 1 hardware report measures:

```
packed bytes per layer-expert object
scale/zero-point metadata bytes
host-to-device copy latency for the complete packed object
first-use/JIT cost separately from steady-state cost
4-bit expert compute latency
slot replacement latency
```

All compared runtime modes use the identical 4-bit profile. The quantizer/backend version, group size, packing layout, model revision, runtime commit, and per-layer expert sizes are written into every evaluation manifest.

Promotion to 8-bit occurs only when:

```
4-bit selected-expert and generated-token parity pass
4-bit reactive, LRU, and predictive modes are reproducible from a clean setup
movement telemetry reconciles all transferred bytes and slot transitions
one-command judge benchmark succeeds on the documented environment
```

### 7A.6 Repository shape

```
expertflow/
  adapters/hf_gemma4.py
  quant/int4_profile.py
  trace/collector.py
  trace/schema.py
  sim/cache_sim.py
  sim/pcie_timeline.py
  runtime/expert_store.py
  runtime/gpu_arena.py
  runtime/cache_manager.py
  runtime/transfer_scheduler.py
  runtime/events.py
  eval/run_suite.py
  eval/baselines.py
  dashboard/
  configs/
  tests/
artifacts/
  sample_trace/
  expected_results/
scripts/
  bootstrap.sh
  smoke_test.sh
  run_replay.sh
  run_live_benchmark.sh
docker/
  Dockerfile
  compose.yaml
pyproject.toml
uv.lock
README.md
REPRODUCING.md
```

## 7B. Judge Reproducibility Contract

The judges must be able to verify the mechanism without reproducing our exact workstation. The repository supports three progressively demanding modes.

### 7B.1 Mode A — installation and collector smoke test

Purpose: verify installation, router observation, schemas, parity checks, and CLI behavior without downloading the full checkpoint.

```
./scripts/bootstrap.sh
uv run expertflow doctor
uv run expertflow smoke --model tiny-random/gemma-4-moe
```

Expected behavior:

```
all router modules discovered
tiny-model prefill/decode events written
hooked and unhooked outputs match within tolerance
trace integrity validator passes
CPU-only tests run where possible; CUDA-specific tests are skipped with a clear reason
```

### 7B.2 Mode B — bundled trace replay and simulator

Purpose: verify the project’s main analysis, policy comparison, evaluator, and dashboard without the full model or a 16 GB GPU.

```
./scripts/run_replay.sh
```

The repository bundles a small, license-safe trace artifact or synthetic-compatible trace fixture containing no private prompt text. The command:

```
validates the trace manifest
replays reactive, static, LRU, session, heuristic, and oracle policies
reproduces a checked-in expected-result manifest within declared tolerances
opens or exports the expert-movement dashboard
clearly labels all replay/simulator numbers as estimated or previously measured
```

### 7B.3 Mode C — full live 4-bit benchmark

Purpose: reproduce the measured Layer 3 claim on compatible NVIDIA hardware.

```
./scripts/run_live_benchmark.sh --config configs/gemma4_4bit_16gb.yaml
```

The command must:

```
run expertflow doctor first
check CUDA capability, VRAM, host RAM, pinned-memory allowance, and free disk
check model authentication/license prerequisites without printing secrets
pin or verify the exact model revision
build/warm the declared 4-bit backend once before timed trials
run reactive, per-layer LRU, and ExpertFlow with identical prompts and settings
write raw events, summary metrics, environment manifest, and exactness report
print actionable failures rather than silently changing precision or model revision
```

A machine that cannot run Mode C must still be able to run Modes A and B.

### 7B.4 Environment locking

The repository provides two supported setup routes:

```
Primary: Python environment from pyproject.toml + uv.lock
Secondary: pinned NVIDIA CUDA container image
```

Lock and record:

```
Python version
PyTorch and CUDA versions
Transformers commit/release
quantization backend and kernel version
Triton or compiled-extension version if used
model and tokenizer revisions
chat-template hash
GPU name, compute capability, driver, VRAM, PCIe link
CPU, RAM, OS, and storage mode
```

Do not depend on a developer’s globally installed packages. Do not silently upgrade the quantization backend.

### 7B.5 CLI and product-flow contract

The public interface follows the closed-loop user journey:

```
expertflow doctor       # inspect hardware, runtime, backend, and model prerequisites
expertflow smoke        # tiny-model hook, schema, and parity validation
expertflow trace        # low-level router-trace collection
expertflow profile      # collect/replay traces and produce a locality/feasibility report
expertflow simulate     # compare cache and PCIe policies under fixed budgets
expertflow recommend    # convert evidence into a machine-specific config
expertflow run          # live reactive/LRU/ExpertFlow execution with telemetry
expertflow benchmark    # repeated same-runtime evaluation suite
expertflow verify       # exactness and causal-comparison report
expertflow replay       # reproduce a prior run without loading the model
expertflow dashboard    # inspect the causal timeline and scorecards
```

The primary judge path is intentionally small:

```bash
expertflow doctor
expertflow replay --bundle demo
expertflow dashboard --run demo
```

The full compatible-hardware path is:

```bash
expertflow profile --model google/gemma-4-26b-a4b-it
expertflow simulate --latest
expertflow recommend --latest
expertflow run --recommended
expertflow verify --latest
```

Every command supports `--help`, emits a machine-readable manifest, and returns a nonzero exit code on validation failure. No command silently changes the model revision, quantization profile, cache budget, or exactness mode.

### 7B.6 Continuous integration

Required public CI:

```
CPU CI: schemas, simulator, policies, CLI, tiny-model hook tests, deterministic replay
GPU CI when available: CUDA event ordering, slot replacement, packed-copy parity, forced-miss correctness
```

The checked-in expected-result manifest includes tolerances rather than brittle exact wall-clock values. Exact token/expert parity is strict; timing comparisons use ranges or relative relationships.

### 7B.7 No hidden demo-only path

The three-minute demo invokes the same public CLI and configuration files documented for judges. No private notebook, uncommitted model patch, hand-edited trace, or unpublished binary may be required for the claimed result.

---

## 7C. Product Experience and Dashboard Contract

### 7C.1 Hardware readiness

`expertflow doctor` reports:

```
GPU and usable VRAM
CPU RAM and safe pinned-memory budget
PCIe generation, negotiated link width, and measured transfer curve
supported 4-bit backend and kernel path
model-profile feasibility
measured/estimated VRAM split: resident model, expert cache, KV/model state, scratch, safety reserve
remaining configurable headroom
blocking incompatibilities and actionable remedies
```

Do not expose raw stack traces as the primary user experience.

### 7C.2 Profile result

`expertflow profile` answers:

```
Is expert routing concentrated or nearly uniform?
How large is the working set per layer?
What are p50/p90 reuse distances?
What cache size captures a useful fraction of demand?
Which layers dominate cold bytes and stalls?
What ready-hit rate is achievable at different expert-cache budgets?
How much VRAM remains for KV/model state or additional sessions at each budget?
Does the oracle show enough headroom to justify runtime work?
```

### 7C.3 Policy laboratory

The dashboard compares policies at equal GPU-slot and prefetch-byte budgets:

```
reactive
static popularity
per-layer LRU
session-frequency/recency
session + global prior
predictor, only if promoted
oracle upper bound

For every policy, also display the paired trade-off:
  expert readiness / blocking stalls
  remaining VRAM headroom for non-expert agent state
```

### 7C.4 Recommendation

`expertflow recommend` must produce a human-readable explanation and a runnable configuration. Each recommendation includes confidence and the evidence that drove it.

Example structure:

```yaml
recommendation:
  decision: enable_predictive_cache
  confidence: medium
  reason:
    - oracle_stall_reduction_is_material
    - session_policy_beats_per_layer_lru
    - p95_reuse_distance_fits_declared_slot_budget
  gpu_slots_per_layer: trace_derived
  victim_slots: trace_derived
  pinned_warm_store_gb: hardware_derived
  reserved_kv_or_model_state_mb: profile_derived
  remaining_uncommitted_vram_mb: measured_or_simulated
  prefetch_horizon_layers: simulator_derived
  speculation: disabled
```

A valid alternative output is `decision: reactive_or_static_is_better` with an explanation.

### 7C.5 Causal timeline

The primary dashboard visual is not a grid of unrelated charts. It is a token/layer timeline that displays:

```
prediction issued
candidate queued
DMA begins
expert becomes GPU-ready
real router demands the expert
expert compute begins
outcome: READY_PREFETCH / LATE_PREFETCH / BLOCKING_MISS / HOT_HIT / VICTIM_HIT
```

The reactive comparison shows the transfer beginning only after demand. The judge should be able to see why a token stalled or did not stall.

The same screen includes a compact stacked VRAM bar:

```
resident backbone / shared components
hot routed experts
victim and prefetch slots
KV or model state
scratch and fragmentation reserve
remaining configurable headroom
```

The bar must use measured values for a live run and estimated values for replay. It may say that remaining headroom is *available* to KV/state or additional sessions, but it must not claim a specific context or concurrency improvement without measuring that runtime profile.

### 7C.6 Trust labels

Every panel and exported scorecard displays:

```
MEASURED or ESTIMATED
model and runtime revision
quantization profile
hardware profile
prompt/evaluation manifest
cache and baseline configuration
exactness status
```

---

## 8. GPU Residency Policy

### 8.1 Keep resident in VRAM when possible

For the Gemma proof of concept:

```
token embeddings if feasible
attention/backbone blocks
routers/gating networks
shared experts
KV cache / attention state
small predictor heads if GPU-side execution is chosen
hot routed experts
victim cache
prefetch buffer
```

For Qwen later, replace the generic attention/KV wording with architecture-specific state:

```
Gated DeltaNet recurrent state
Gated Attention state
model-specific recurrent snapshots
router state
shared experts
```

### 8.2 Store in CPU RAM

```
all quantized routed experts
cold expert pages
warm expert pages in pinned memory
backup full expert store
trace buffers
prior tables
```

### 8.3 Do not use SSD in active decode path

SSD is acceptable for loading the model into CPU RAM before serving. SSD must not be used as the normal source for expert pages during token generation.

---

## 9. Exactness Principle

The exactness rule is non-negotiable:

```
The true router is authoritative.
Predictors never override final expert choice.
```

Allowed in exact mode:

```
predict experts
prefetch experts
prioritize experts
cache experts
evict experts
bias draft proposals toward loaded-expert-friendly paths
fallback to blocking load on miss
```

Not allowed in exact mode:

```
skip a cold expert selected by the real router
replace real router top-k with predictor top-k
verify speculative tokens using only currently loaded experts
permanently disable experts because the conversation appears English-only
```

If an approximate mode is implemented later, it must be explicitly labeled and measured separately.

---

## 10. Two-Table Design

The cache policy uses two separate tables:

```
1. Global prior table — learned from offline traces, useful for cold start.
2. Session table — built from scratch for the current conversation.
```

The global prior prevents empty-cache startup. The session table prevents a bad global prior from dominating the current session.

### 10.1 Global prior table

```rust
struct GlobalPriorRecord {
    model_id: String,
    quant_profile: String,
    domain_id: DomainId,
    layer_id: u16,
    expert_id: u16,
    mean_usage_rate: f32,
    p50_reuse_distance: f32,
    p90_reuse_distance: f32,
    coactivation_sketch_id: u32,
    confidence: f32,
    last_refreshed_epoch: u64,
}
```

Domain-conditioned prior tables:

```
global_prior[english_chat]
global_prior[code]
global_prior[math]
global_prior[translation]
global_prior[multilingual]
global_prior[long_reasoning]
global_prior[tool_use]
global_prior[vision_text]   // schema-reserved, disabled in text-only MVP
```

MVP may start with only: `general_chat`, `code`, `math`, `translation`.

#### 10.1.1 Coactivation representation for Gemma MVP

Use a concrete per-layer dense coactivation matrix rather than an underspecified foreign-key sketch:

```rust
struct LayerCoactivationMatrix {
    model_id: String,
    quant_profile: String,
    domain_id: DomainId,
    layer_id: u16,
    // [expert_i][expert_j] count or decayed score
    pair_score: [[f32; 128]; 128],
}
```

Memory:

```
128 × 128 × 4 bytes = 64 KB per layer
64 KB × 30 layers ≈ 1.9 MB per domain table
```

Usage:

```
if expert A is hot and A frequently coactivates with expert B:
    raise B's prefetch/admission priority

if expert B has high coactivation but low recent usage:
    load B only when spare prefetch budget exists
```

Domain blending policy for MVP: use argmax domain from the `DomainMix` to select a single coactivation matrix. Do not weight-average matrices across domains in MVP. Revisit after simulator results.

Later: compress into count-min sketches, sparse top-M neighbor lists, or low-rank factorization if storage or lookup cost grows.

### 10.2 Session table

```rust
struct SessionExpertRecord {
    layer_id: u16,
    expert_id: u16,
    last_used_token: u64,
    recent_count: u16,
    // Exponentially decayed recent usage score.
    // Decay half-life: 32 tokens (tunable via config).
    // At each token step: score *= 0.5^(1/32); +1.0 on use.
    decayed_recent_score: f32,
    total_session_count: u32,
    cache_hits: u32,
    cache_misses: u32,
    predicted_prob: f32,
    actual_router_prob_ema: f32,
    // Bucketed reuse distance tracking.
    // See bucket definitions in section 10.2.1.
    reuse_distance_short_bucket: u8,
    reuse_distance_long_bucket: u8,
    reuse_distance_p50: f32,
    reuse_distance_p90: f32,
    topic_phase_id: u16,
    resident_state: ResidentState,
}
```

Session table roles:

```
measure actual current expert usage
judge whether the global prior is helping
identify topic shifts
drive expert residency after prefill
provide evidence for per-layer quota rebalance
```

#### 10.2.1 Reuse distance buckets

Reuse distance is heavy-tailed. Do not use a single fixed-decay EMA.

```
bucket 0: reused within 1 token
bucket 1: reused within 2–4 tokens
bucket 2: reused within 5–16 tokens
bucket 3: reused within 17–64 tokens
bucket 4: reused within 65–256 tokens
bucket 5: reused after >256 tokens or not yet reused
```

The simulator exports p50/p90 reuse distance per (layer, expert, domain) and uses those percentiles for victim-cache sizing and admission thresholds.

### 10.3 Effective score: MVP formula

```
effective_score =
    0.55 × session_score
  + 0.30 × domain_prior_score
  + 0.15 × predictor_score
  - eviction_penalty
```

Where:

```
session_score       = decayed_recent_score + hit/miss usefulness signal
domain_prior_score  = global prior score for argmax current domain
predictor_score     = near-future probability from the active predictor
eviction_penalty    = high if expert is pinned, loading, or very recently used
```

Coefficients are defaults. Tune only after the Layer 1 simulator produces traces.

### 10.4 Effective score: later formula

```
effective_score =
    α × global_prior_score
  + β × session_recent_score
  + γ × predictor_score
  + δ × current_prefetch_urgency
  + ε × coactivation_score
  - λ × memory_cost
  - μ × transfer_cost
  - ν × pollution_risk
```

Do not use the expanded formula until trace data proves each term adds value.

### 10.5 α/β blending schedule

```
startup:          α=0.70, β=0.30
after prefill:    α=0.40, β=0.60
stable session:   α=0.10–0.20, β=0.80–0.90
topic shift:      temporarily raise α
```

Hard floor: `α_min = 0.10`. The global prior remains useful during topic shifts when the session table is stale.

### 10.6 Prior-vs-session comparison

Track continuously:

```
prior-only hit rate
session-only hit rate
merged-policy hit rate
misses avoided per MB loaded
wasted resident experts from prior
wasted resident experts from session
```

If the prior is weak for this session: lower α, increase session quota, stop pinning prior experts aggressively.

If the session table is sparse or a topic shift is detected: raise α temporarily, use domain prior to restabilize.

---

## 11. Domain Classifier

The domain classifier affects which global prior tables are active, which predictor heads run, topic-shift detection, α/β schedule adjustments, prefetch aggressiveness, and MTP/draft window size.

### 11.1 Classifier output

```rust
struct DomainMix {
    english_chat: f32,
    code: f32,
    math: f32,
    translation: f32,
    multilingual: f32,
    long_reasoning: f32,
    tool_use: f32,
    vision_text: f32, // always hard zero-clamped in text-only MVP
}
```

In text-only MVP, `vision_text` must be hard zero-clamped regardless of classifier output. The classifier may have seen image-description text in training data and could fire a nonzero score on text-only prompts.

Example output:

```
english_chat: 0.65
code: 0.25
math: 0.10
vision_text: 0.00  // clamped
```

### 11.2 MVP classifier features

```
token n-gram features from recent prompt/decode window
language ID signal
presence of code fences / indentation / syntax tokens
math symbols and equation density
translation markers
router entropy summary from prefill
expert diversity summary from first N tokens
recent topic-shift score
```

Avoid hidden-state hooks for MVP.

### 11.3 Classifier architecture

MVP:

```
logistic regression or tiny MLP
runs every 32 decode tokens by default
CPU-side acceptable if latency is negligible
```

Later:

```
router-logit feature MLP
small gradient-boosted model
distilled classifier from trace labels
```

### 11.4 Update frequency

```
run during prefill
run every 32 decode tokens
run immediately on large router-distribution shift
run immediately when user sends a new message
```

### 11.5 MVP training and update policy

The MVP domain classifier is trained offline on labeled prompt examples and trace metadata. It does not perform online parameter updates during inference.

Allowed at runtime:

```
recompute domain mixture from recent text
recompute domain mixture after a new user message
adjust cache policy based on classifier output
record misclassification and health metrics
```

Not allowed in MVP:

```
update classifier weights online
fine-tune classifier during live inference
let the classifier permanently override session evidence
```

Training data must include mixed cases: English chat with embedded code, code with natural-language explanation, math in prose, translation embedded in chat, multilingual code comments, long reasoning without formal math notation.

### 11.6 Latency budget

```
< 50 microseconds per call for CPU-side simple model
< 10 microseconds if GPU-side and called frequently
```

If the classifier exceeds its budget, fall back to session table + general prior only.

---

## 12. Predictor System

### 12.1 Predictor goal

For layer L and upcoming token position T, which experts are likely to be needed? Predictors do not generate text and do not replace the real router.

### 12.2 Gated mixture-of-predictors

Use the domain classifier to activate at most 1–2 predictor heads per token. Do not run all heads every token.

```
Domain: english_chat + code
Active: general predictor, code predictor
Inactive: math, translation, multilingual
```

### 12.3 Concrete MVP predictor architecture

Use a small linear classifier over router-derived features first, not hidden states.

Rationale:

```
router logits/top-k are already computed
router features are cheaper and more interpretable than hidden states
hidden-state access requires architecture-specific hooks
```

MVP input features:

```
current layer id
previous layer top-k experts
previous layer router probabilities
recent expert usage bitmap/sketch for this layer
recent expert usage bitmap/sketch for neighboring layers
domain mixture vector
position bucket
prefill/decode mode flag
MTP/draft token offset if applicable
```

MVP output: ranked list of experts per layer with probability or confidence per expert.

MVP model choices: per-layer linear classifier, low-rank factorized classifier, small MLP with one hidden layer. Avoid transformer predictors in MVP.

### 12.4 Predictor ensemble and correction oversight

```
Predictor A: general router-logit predictor
Predictor B: domain-specific predictor
Session table: current empirical hot set
Global prior: cold-start/domain prior
Real router: final authority
```

Policy:

```
predictors + session table agree: prefetch first
predictors disagree: load only if budget allows
real router disagrees: blocking load, update miss stats, penalize bad predictor
```

Agreement affects priority only, not correctness.

### 12.5 Predictor health metrics

Do not optimize raw top-k accuracy alone:

```
useful_prefetch_rate = correctly_prefetched_experts / total_prefetched_experts
misses_avoided_per_MB_loaded
misses_avoided_per_ms_predictor_latency
cache_pollution_rate
predictor_agreement_quality
```

A predictor that improves top-k accuracy but loads too many wrong experts is harmful.

---

## 13. Cache Design

### 13.1 Working set

Track the current conversation's active expert working set: experts used often, experts that are recent, experts that recur after short reuse distances, one-off noise experts.

### 13.2 Cache hierarchy

```
GPU hot cache
GPU victim cache
GPU prefetch buffer
pinned CPU warm store
CPU cold store
SSD emergency backing store
```

### 13.3 Cache unit

One expert in one layer = one cache object for MVP. Later: shard experts if full expert pages are too large.

### 13.4 Replacement policy

MVP: LRU + decayed LFU hybrid.

Later: 2Q / ARC-like separation of one-hit vs recurring experts, TinyLFU-style admission, cost-aware eviction, reuse-distance-aware eviction.

### 13.5 Admission control

Admit into hot cache if:

```
used more than once recently
or predicted to recur soon
or appears repeatedly during prefill
or belongs to selected global/domain prior
or co-activates with already-hot experts
```

Otherwise: load temporarily, use once, do not promote to hot cache.

### 13.6 Per-layer partitioning

Expert cache must be per-layer aware. Expert 20 in layer 4 is unrelated to expert 20 in layer 30. Reserve a base quota per layer, then allow adaptive redistribution.

### 13.7 Victim cache

Stores recently evicted experts that may bounce back during routing oscillation or topic shifts. Do not hardcode a fixed size before tracing.

Sizing rule:

```
victim_cache_size = f(p75/p90 reuse distance, expert object size, oscillation rate)
```

Layer 1 simulator sweep: 128 MB, 256 MB, 512 MB, 1 GB, 2 GB. Choose the smallest size that captures most bounce-back misses.

### 13.8 Thrashing detection

Detect thrashing when:

```
miss rate spikes
same experts are repeatedly evicted and reloaded
prefetch queue backs up
MTP window increases unique cold expert union too much
cache hit rate drops below threshold for N tokens
```

Responses:

```
reduce MTP/draft window
pin stable session experts
lower prefetch aggressiveness
increase admission threshold
increase per-layer quota for noisy layers
reduce global-prior slots if prior is wasting space
```

---

## 14. PCIe Bandwidth Feasibility Model

This is load-bearing. Do not proceed to runtime engineering until the offline simulator estimates whether expert transfers can be hidden.

### 14.1 Transfer equation

```
transfer_time_ms = cold_transfer_MB / effective_bandwidth_MB_per_ms + overhead_ms
```

Planning numbers for modeling:

```
PCIe 4.0 x16 theoretical: ~32 GB/s
practical sustained DMA: ~20–28 GB/s
MVP planning number: 25 GB/s = 25 MB/ms
```

### 14.2 Required cache hit rate

If no-cache routed expert traffic is X MB/token (X is a placeholder until Layer 1 measures actual expert-object sizes):

```
cold_MB_per_token = X × (1 - H)
transfer_ms_per_token ≈ cold_MB_per_token / 25

80% hit rate: 0.20X / 25 ms
90% hit rate: 0.10X / 25 ms
95% hit rate: 0.05X / 25 ms
```

These numbers are before launch overhead, scheduling overhead, quantization metadata, pinned-memory overhead, and imperfect overlap.

X must be replaced by measured Gemma expert-object sizes from Layer 1 tensor inspection. Do not use Qwen-derived sizing heuristics to estimate Gemma traffic.

### 14.3 Gemma traffic must be measured

Layer 1 first hardware task: inspect actual tensor shapes under each candidate runtime profile and compute:

```
expert object size per layer
shared expert size
router size
backbone size
quantization scale/metadata overhead
```

Do not commit to a specific expert cache budget until the non-expert VRAM footprint is measured.

### 14.4 Pipeline model

The simulator must model:

```
GPU compute time per layer
expert transfer time per layer
copy/compute overlap
prefetch queue depth
pinned memory staging
cache hit/miss sequence
MTP/draft window size
unique cold expert union per verification pass
```

High cache hit rate is not sufficient if transfers arrive too late.

### 14.5 Feasibility thresholds

Proceed to Layer 3 only if traces and the event-driven simulator satisfy the Layer 1 go/no-go gate, including at least one of:

```
cache hit rate >= 90% for stable single-domain sessions
or transfer stalls can be hidden behind compute/prefill/speculation
or throughput improvement over naive CPU offload >= 1.5×
```

Stretch target: >= 2× throughput over naive reactive offload.

---

## 15. VRAM Budgeting

### 15.1 Budget components

```
non-expert backbone weights
routers
shared experts
embeddings/lm_head if resident
KV cache or model-specific state
activation scratch buffers
quantization metadata
hot routed expert cache
victim cache
prefetch buffer
runtime fragmentation margin
```

### 15.2 Rough prior estimate before measurement

```
25.2B params × 4 bits ≈ 12.6 GB raw 4-bit weight payload
+ quantization metadata
+ alignment/fragments
+ runtime buffers
+ KV/state
```

Gemma has 128 total routed experts with only 8 active per layer, so a large fraction of total parameters is routed expert storage. A plausible Layer 1 planning hypothesis:

```
text-only non-expert/shared/router/embedding/runtime: ~3–5 GB at 4-bit
routed expert storage: majority of remaining weight payload, in CPU RAM
initial available VRAM for expert cache: unknown, possibly 2–6 GB depending on context length and runtime format
```

This is a hypothesis only. Replace with measured tensor accounting before Layer 3.

### 15.3 Required Layer 1 VRAM table

Layer 1 must produce:

```
component                                  MB
----------------------------------------------
text backbone weights                         ?
routers                                       ?
shared experts                                ?
embeddings/lm_head                            ?
vision encoder (optional, excluded MVP)       ?
KV/state at context 2K                         ?
KV/state at context 8K                         ?
scratch buffers                               ?
quant metadata                                ?
fragmentation reserve                         ?
available for routed expert cache             ?
reserved KV/model-state headroom                ?
remaining configurable VRAM headroom            ?
```

### 15.4 Cache budget policy

If available expert-cache budget is B:

```
session hot cache:   55–75% of B
global-prior cache:  10–30% of B
victim cache:        trace-derived
prefetch buffer:     5–15% of B
```

Actual splits must be simulator-driven.

### 15.5 Agent-state headroom reporting

For every simulated recommendation and live run, report:

```
peak resident non-expert model memory
peak routed-expert cache allocation
victim/prefetch allocation
peak KV or architecture-specific model state at the tested context
scratch and fragmentation reserve
remaining configurable VRAM headroom
```

The product-level interpretation is:

> **Memory not permanently committed to the full routed-expert set becomes a configurable budget that a compatible local-agent runtime may use for KV/model state, longer active trajectories, or more sessions.**

This is a memory-accounting statement, not a context-length result. Do not convert headroom into token or session counts unless the relevant runtime is tested at those settings.

---

## 16. Speculative Decoding / MTP Interaction

### 16.1 Why speculation helps

Lookahead from speculative decoding or MTP can hide PCIe transfers: while the GPU verifies current draft tokens, the runtime prefetches experts likely needed next.

### 16.2 Why speculation hurts

Verifying multiple tokens may require the union of experts across all token positions. Cold expert union can explode. PCIe queue backs up.

### 16.3 Control by cold expert union, not token count

```
if predicted cold expert union is small: allow larger window
if predicted cold expert union is large: shrink window
```

### 16.4 Dynamic window policy

```
cache hit rate high and acceptance rate high: increase window to 2–4
cache hit rate low or expert diversity high: reduce window to 1–2
thrashing detected: disable speculation or use window 1
```

For Gemma MVP: start with no speculation or external draft window = 1. Only add speculation after baseline caching beats reactive offload.

For Qwen later: use native MTP only after DeltaNet/state handling is understood.

### 16.5 Partial prefix acceptance

If draft proposes t1, t2, t3, t4 and t1/t2 are hot but t3 requires many cold experts:

```
accept t1, t2
stop before t3
prefetch missing experts
resume with smaller window
```

### 16.6 Loaded-expert-aware drafting

```
Draft can be cache-aware.
Verifier must be exact.
```

The draft path may prefer tokens likely to use loaded experts. The verifier must never accept tokens using only currently loaded experts.

---

## 17. Cache Object Lifecycle

```rust
enum ResidentState {
    CPU_COLD,        // normal pageable or mmap-backed CPU storage
    CPU_WARM_PINNED, // pinned host staging/cache, ready for fast DMA
    GPU_PREFETCHING, // async copy in flight
    GPU_HOT,         // resident in GPU expert cache
    GPU_VICTIM,      // recently evicted, briefly kept in GPU memory
    GPU_PINNED,      // protected from eviction for correctness or thrash recovery
    EVICTING,        // being moved out of GPU cache
}
```

`CPU_WARM_PINNED` is a host-side staging/cache level, not just a transient label. Experts move from `CPU_COLD` to `CPU_WARM_PINNED` when:

```
they are likely needed soon but GPU space is unavailable
they are repeatedly missed but not hot enough for VRAM residency
they are selected for a near-future prefetch batch
they were recently evicted from GPU and may bounce back
```

The scheduler prefers DMA from `CPU_WARM_PINNED` over pageable `CPU_COLD`. The pinned budget must be capped to avoid harming system RAM performance. See Section 30 for the budget config field.

Lifecycle:

```
CPU_COLD
  -> CPU_WARM_PINNED
  -> GPU_PREFETCHING
  -> GPU_HOT
  -> GPU_VICTIM
  -> CPU_WARM_PINNED or CPU_COLD
```

Pinned experts include: shared experts, currently executing layer experts, experts needed by already-scheduled verifier passes, experts protected during thrash recovery.

`CPU_WARM_PINNED` TTL policy: if a warm-pinned expert is not scheduled for transfer within a TTL window (default: 128 decode tokens, tunable), demote to `CPU_COLD` and release the pinned page.

---

## 18. Prefetch Candidate Schema

```rust
enum PredictionSource {
    TrueRouterRequired,    // blocking load, miss already occurred
    SessionTableHot,       // high session score
    GlobalPriorDomain,     // domain prior recommendation
    LinearPredictor,       // linear router-feature predictor
    CoactivationInference, // coactivation matrix recommendation
    SpeculativeOnly,       // speculative window, lowest priority
}

struct PrefetchCandidate {
    model_id: String,
    layer_id: u16,
    expert_id: u16,
    token_offset: u8, // max lookahead 255 tokens; intentional MVP cap
    source: PredictionSource,
    probability: f32,
    expected_value: f32,
    transfer_cost_bytes: u32,
    deadline_us: u64,
    is_agreed_by_multiple_predictors: bool,
}
```

Expected value:

```
expected_value =
    P(expert needed)
  × P(token accepted if speculative)
  × miss_cost_saved
  × urgency
  ÷ transfer_cost
```

---

## 19. Transfer Scheduler

The PCIe scheduler should:

```
use pinned host memory
batch adjacent expert transfers when possible
avoid tiny fragmented transfers
use async DMA / CUDA streams
prioritize experts with earliest deadlines
avoid loading experts likely to be evicted immediately
track copy queue depth
track copy/compute overlap
```

Priority order:

```
1. TrueRouterRequired — blocking miss
2. expert required by scheduled exact verifier pass
3. SessionTableHot + predictor agreement
4. GlobalPriorDomain prefetch
5. CoactivationInference prefetch
6. SpeculativeOnly — drop first under pressure
```

Source tracking is mandatory. The runtime must report prefetch usefulness broken down by `PredictionSource` so bad sources can be downweighted rather than globally reducing prefetching.

`CPU_WARM_PINNED` priority behavior:

```
if expert is likely soon but cannot fit in VRAM:
    promote CPU_COLD -> CPU_WARM_PINNED if pinned budget allows

if expert is in CPU_WARM_PINNED and becomes urgent:
    schedule GPU_PREFETCHING before equivalent CPU_COLD candidates

if CPU_WARM_PINNED expert is unused beyond its TTL:
    demote to CPU_COLD
```

---

## 20. Runtime Loop

### 20.1 Prefill

```
for each prompt token:
    run model normally
    observe true router expert choices
    update session table
    update reuse-distance bucket stats
    update domain classifier summary
```

After prefill:

```
build initial session hot set
compare with global prior
rebalance per-layer quotas
promote high-confidence session experts to GPU_HOT
```

### 20.2 Decode

```
for each generated token:
    classify domain every 32 decode tokens (default)
    compute effective expert scores
    prefetch likely next experts
    run true model router
    if selected experts are resident: continue
    else: blocking load, update miss stats
    update session table
    update predictor health
    adjust MTP/draft window
    evict low-score non-pinned experts
```

### 20.3 Topic shift detection

Reference distributions:

```
P_recent: smoothed expert/domain distribution over latest 32–128 tokens
P_session_ema: exponentially weighted running distribution over current topic phase
shift_score: Jensen-Shannon divergence between P_recent and P_session_ema
```

Do not compare against only the first-N-token prefill distribution. That over-triggers on gradual topic evolution.

Trigger a topic shift when `shift_score` exceeds a threshold, language classifier changes, code/math/translation signals appear, expert hit rate collapses, or a new user message arrives with a different domain signature.

When a shift is confirmed:

```
assign new topic_phase_id
raise α temporarily
reduce MTP window
decay old session records faster
increase prefetch exploration budget
```

---

## 21. Per-Layer Quotas

Start with equal base quota:

```
cache_slots_per_layer = total_slots / number_of_layers
```

Adapt by layer expert diversity, miss cost, reuse distance, thrashing rate, and predictor confidence.

```
Layer 7: stable, low diversity -> smaller quota
Layer 18: high reuse, high miss cost -> larger quota
Layer 24: noisy one-off experts -> stricter admission, not necessarily larger quota
```

Do not allow one noisy layer to evict useful experts from all other layers.

---

## 22. Global Prior Training Pipeline

### 22.1 MVP offline pipeline

```
1. Select representative prompt sets.
2. Run Gemma 4 26B A4B with tracing enabled.
3. Record router choices per token/layer.
4. Compute expert usage frequencies.
5. Compute reuse-distance distributions.
6. Compute coactivation matrices.
7. Build domain-conditioned prior tables.
8. Validate against held-out traces.
```

Prompt categories: English chat, code generation, code debugging, math reasoning, translation, multilingual chat, long-context summarization. Vision-text only after text-only MVP is proven.

### 22.2 Known production gap

Deployment-time prior refresh is not solved in MVP. Open questions:

```
How often should priors be refreshed?
Should user sessions contribute anonymized traces?
How is domain drift detected globally?
How are regressions prevented when updating priors?
How do quantization/runtime changes affect prior validity?
```

For MVP, global priors are static artifacts generated offline.

---

## 23. Layer 1A Router Trace Collector

This is the first real deliverable.

### 23.1 Purpose

```
Is there enough expert locality to exploit?
What is the reuse-distance distribution?
How large should the victim cache be?
How large is the hot working set per layer?
How much VRAM remains for expert cache after non-expert components?
How bad is naive reactive offload?
```

### 23.2 Trace fields

```rust
struct ExpertTraceEvent {
    request_id: u64,
    token_index: u64,
    phase: PrefillOrDecode,
    layer_id: u16,
    expert_ids: [u16; MAX_TOPK],
    router_probs: [f32; MAX_TOPK],
    domain_mix: DomainMix,
    was_resident: bool,
    simulated_transfer_bytes: u32,
    timestamp_us: u64, // router observation time; movement timing uses ExpertMovementEvent
}
```

### 23.3 Layer 1 trace outputs

```
per-layer expert frequency histograms
reuse-distance CDFs and bucket distributions
expert coactivation matrices
cache hit-rate curves by cache size
victim-cache benefit curves
prefetch oracle upper bound
PCIe transfer simulation
VRAM budget component table
topic-shift traces
```

Write the offline simulator before touching GPU cache runtime integration.

---

## 23A. Gemma 4 Router Trace Collection Solution

### 23A.1 Why a forward hook works

In the Hugging Face Gemma 4 implementation, every MoE-enabled text decoder layer owns a `Gemma4TextRouter`. The router computes probabilities across all experts, selects top-k experts, normalizes selected weights, and returns:

```python
router_probabilities, top_k_weights, top_k_index
```

The decoder then passes `top_k_index` and `top_k_weights` to the expert collection. A read-only forward hook on each router therefore observes the actual selected experts before expert computation without changing the router's result.

### 23A.2 Required collector behavior

The collector must:

```
find every module whose class is Gemma4TextRouter
map its module path to the decoder layer ID
register a read-only forward hook
capture router probabilities, selected expert IDs, and selected weights
map flattened router rows back to absolute token positions
separate prefill events from decode events
write events incrementally instead of holding the dataset in RAM
remove hooks cleanly after every run
fail loudly if router output shape or module names change
```

### 23A.3 Do not rely on hidden `generate()` internals

For the reference collector, use an explicit batch-size-1 generation loop:

```
1. Apply the official chat template.
2. Run one prefill forward pass over the complete prompt.
3. Record all prompt token/layer routing events.
4. Select the next token.
5. Feed one generated token with past_key_values.
6. Record one decode routing event per MoE layer.
7. Repeat until EOS or max_new_tokens.
```

This makes token-to-router-event alignment explicit. A later high-throughput collector may instrument `generate()` or a serving runtime, but only after parity tests.

### 23A.4 Two-stage validation

Stage A — schema and hook validation:

```
model: tiny-random/gemma-4-moe
purpose: verify module discovery, hook output order, shapes, token mapping,
         prefill/decode labeling, and JSON/Parquet writing
```

Stage B — target collection:

```
model: google/gemma-4-26B-A4B-it
purpose: produce authoritative Gemma traces
runtime: pinned Transformers commit and model revision
```

The tiny-random model is not useful for learned routing behavior. It is only a cheap integration test.

### 23A.5 Reference implementation

The product repository should include:

```
tools/trace_gemma4_router.py
```

The companion file produced with this spec is:

```
expertflow_gemma4_router_trace_collector.py
```

The reference implementation:

```
uses AutoProcessor and Gemma4ForConditionalGeneration
supports JSONL conversations with OpenAI-style messages
registers hooks by router class name instead of a brittle fixed model path
runs manual prefill and decode forwards
stores selected expert IDs and selected weights
optionally stores full router probabilities
writes a reproducibility manifest with hashes and software versions
supports an optional 4-bit loading request when the installed stack supports it
```

It is a scaffold, not a claim that the full 26B checkpoint was tested in this document-generation environment.

### 23A.6 Runtime and hardware strategy for trace generation

Trace generation does not have to occur on the final 16 GB deployment GPU.

Recommended order:

```
1. Validate collector with tiny-random Gemma locally.
2. Run a small BF16/FP16 Gemma trace sample on rented 48–80 GB GPU capacity.
3. Validate routing stability against the intended quantized deployment path.
4. Collect the main trace corpus using the closest practical deployment configuration.
5. Recollect a calibration subset whenever checkpoint, quantization, router precision,
   tokenizer, chat template, or runtime changes.
```

Because quantized expert outputs affect later hidden states, they can indirectly affect routing in later layers and tokens. Predictor artifacts must therefore be bound to a declared quantization/runtime profile.

### 23A.7 Conversation replay modes

Support three modes:

**Mode 1 — full-context replay**

```
Feed an existing multi-turn conversation as the prompt.
Generate a continuation.
Fastest way to collect broad prefill and decode traces.
```

**Mode 2 — turn-by-turn reconstruction**

```
Feed each user turn in sequence.
Let Gemma generate each assistant turn.
Preserve the accumulated conversation state.
Preferred for realistic session-cache and topic-shift research.
```

**Mode 3 — scripted topic-shift scenarios**

```
Construct conversations that intentionally move between chat, code, math,
translation, multilingual text, and tool-like structured outputs.
Required for testing prior/session rebalancing.
```

The Layer 1 pilot may begin with Mode 1. A later research-grade corpus should also include Mode 2 and Mode 3.

---

## 23B. Canonical Trace Dataset Design

### 23B.1 Trace manifest

Every collection shard must include a manifest:

```rust
struct TraceManifest {
    schema_version: String,
    trace_dataset_id: String,
    model_id: String,
    model_revision: String,
    model_config_hash: String,
    tokenizer_revision: String,
    chat_template_hash: String,
    transformers_commit: String,
    runtime_commit: String,
    quantization_profile: String,
    router_precision: String,
    expert_precision: String,
    batch_size: u16,
    sampling_config: SamplingConfig,
    prompt_dataset_ids: Vec<String>,
    prompt_dataset_revisions: Vec<String>,
    split_manifest_hash: String,
    hardware_profile: HardwareProfile,
    collector_config_hash: String,
    created_at_epoch: u64,
}
```

A predictor must refuse to load if its feature-schema hash or target model/runtime compatibility declaration does not match.

### 23B.2 Request record

```rust
struct TraceRequest {
    request_id: String,
    conversation_id: String,
    source_dataset: String,
    source_record_id_hash: String,
    split: DataSplit,
    domain_labels: DomainMix,
    message_count: u16,
    prompt_token_count: u32,
    generated_token_count: u32,
    seed: u64,
    temperature: f32,
    top_p: f32,
    top_k_sampling: u32,
    max_new_tokens: u32,
    finish_reason: String,
}
```

Raw prompt text should be stored separately from routing events and may be omitted entirely from publishable trace artifacts.

### 23B.3 Router event record

```rust
struct RouterTraceEvent {
    schema_version: String,
    request_id: String,
    conversation_id: String,
    turn_index: u16,
    phase: PrefillOrDecode,
    forward_id: u64,
    hook_order: u64,
    token_index: u64,
    token_id: u32,
    layer_id: u16,
    selected_expert_ids: [u16; TOP_K],
    selected_expert_weights: [f16; TOP_K],
    router_top_m_expert_ids: Vec<u16>,
    router_top_m_probabilities: Vec<f16>,
    full_router_probabilities: Option<Vec<f16>>,
    feature_available_order: u64,
    target_required_order: u64,
}
```

### 23B.4 Storage tiers

**Core tier — always collect**

```
request/conversation IDs
phase
absolute token index
input token ID
layer ID
selected expert IDs
selected expert weights
model/runtime manifest
```

Use for:

```
cache simulation
reuse-distance analysis
coactivation matrices
session/global priors
basic multi-label training
```

**Enhanced tier — collect on training shards**

```
router top-M IDs and probabilities, recommended M=16 or 32
router entropy
previous-layer and previous-token summaries
feature availability ordering
```

Use for linear/MLP predictor training and calibration.

**Debug tier — limited subset only**

```
full 128-way router probability vector
selected hidden-state summaries or compressed projections
precise CUDA timing events
full text and decoded tokens when licensing/privacy allows
```

Do not store full hidden states by default. They are expensive, can leak text information, and are not part of the MVP predictor plan.

### 23B.5 Preferred physical format

```
Collection output: append-only JSONL for simplicity and crash recovery.
Canonical training format: partitioned Parquet with Zstandard compression.
Partition keys: model_profile / split / domain / phase / shard_id.
```

Keep request metadata and router events in separate tables joined by `request_id`.

### 23B.6 Approximate storage planning

Gemma produces one router event per MoE layer per processed token. With 30 MoE layers:

```
100,000 processed tokens -> approximately 3 million router events
1,000,000 processed tokens -> approximately 30 million router events
```

Full 128-way float16 probabilities add 256 raw bytes per event before metadata and compression:

```
3 million events -> ~768 MB raw probabilities alone
30 million events -> ~7.68 GB raw probabilities alone
```

Therefore:

```
collect core fields for every event
collect top-M probabilities for most training data
collect full probabilities only for a calibration/debug subset
```

### 23B.7 Integrity checks

For every request:

```
router-event layer IDs must match discovered MoE layers
selected expert count must equal configured top-k
expert IDs must be in [0, num_experts)
prefill event rows must equal prompt_tokens × MoE_layers
decode event rows must equal processed_decode_tokens × MoE_layers
hook order must be strictly increasing
no duplicate (request, forward, token, layer) events
request summary token counts must match trace counts
```

Randomly replay at least 1% of requests and confirm identical selected experts under deterministic settings.

---

## 23C. Prompt Corpus and Sampling Plan

### 23C.1 Dataset strategy

Use a mixture rather than one dataset. No single dialogue corpus provides adequate code, math, multilingual, tool-use, long-context, and topic-shift coverage.

Initial candidate pool:

| Category | Candidate source | Purpose |
|---|---|---|
| Natural multi-turn chat | Pure-Dove, OpenAssistant | Session locality and ordinary dialogue |
| Broad synthetic conversation | UltraChat 200k, Capybara | Topic diversity and scale |
| Real-world messy interactions | WildChat non-toxic/filtered releases | Ambiguity, code-switching, topic shifts |
| Code | Curated public code instruction/evaluation prompts | Domain specialization and structured syntax |
| Math/reasoning | Curated math and reasoning prompts | Long reasoning and expert diversity |
| Translation/multilingual | Parallel and multilingual instruction prompts | Language shifts and mixed-language routing |
| Tool/JSON | Function-calling and structured-output prompts | Syntax-heavy behavior and constrained decoding |
| Topic-shift scripts | ExpertFlow-authored scenarios | Explicit cache invalidation and prior recovery tests |

### 23C.2 MVP corpus composition hypothesis

Start with a balanced research corpus rather than mirroring internet frequency:

```
25% general English multi-turn chat
20% code generation/debugging/explanation
15% math and formal reasoning
10% translation
10% multilingual and code-switching
10% long-context summarization/retrieval
5% tool use / JSON / structured outputs
5% scripted abrupt and gradual topic shifts
```

This is an experiment plan, not a final production distribution. Reweight after observing routing diversity and target use cases.

### 23C.3 Sampling within conversations

Include:

```
short: 1–2 turns
medium: 3–8 turns
long: 9+ turns
short context: <=2K tokens
medium context: 2K–8K tokens
longer research subset: 8K–32K tokens
```

Do not begin with 256K contexts. Long context materially increases collection cost and is not necessary to establish basic routing locality.

### 23C.4 Generated-token policy

For every source conversation, trace:

```
prefill over the available conversation context
at least 32–128 generated tokens when feasible
longer generation for reasoning/code subsets
multiple deterministic seeds for a small variance study
```

The majority of training requests should use one fixed sampling profile for comparability. A smaller robustness subset should vary temperature and top-p.

### 23C.5 Data split policy

Split by complete conversation before trace generation where possible:

```
train: 80%
validation: 10%
test: 10%
```

Never randomly split token events from the same conversation across sets.

Additional frozen evaluations:

```
out-of-source test: prompt source absent from training
leave-one-domain-out tests
abrupt topic-shift test
long-session test
rare-expert-heavy test
multilingual/code-switch test
quantization/runtime transfer test
```

### 23C.6 Privacy and safety

```
Do not collect private user conversations without explicit consent.
Hash source record IDs.
Store raw text separately with stricter access controls.
Allow trace-only export with no raw text.
Run PII and secret scanning before retaining or publishing text.
Respect source dataset licenses and removal requests.
Document whether generated Gemma outputs inherit source-distribution restrictions.
```

Routing traces can still leak properties of the original text. Treat them as derived data, not automatically anonymous data.

---

## 23C-FAST. 24-Hour Locality Dry Run

The first empirical result is not a learned predictor. It is a fast, stratified answer to: **does Gemma expose a cacheable working set under the target runtime profile?**

### 23C-FAST.1 Minimum probe corpus

Use 24–40 independent conversations, not only five near-duplicates:

```
6–8 stable general-chat sessions
6–8 code generation/debugging sessions
6–8 math or structured-reasoning sessions
4–8 translation, multilingual, structured-output, or scripted topic-shift sessions
```

Use deterministic decoding for the core probe, 64–128 generated tokens where feasible, and the same model revision, chat template, router precision, expert precision, and quantization path planned for the live run. Include prompt/prefill routing because prefill may establish the initial hot set, but report prefill and decode separately.

### 23C-FAST.2 Required next-day evidence bundle

By 2026-07-15, produce:

```
full-model or declared fallback load result
peak and component VRAM accounting
router-hook parity and trace-integrity result
per-layer normalized entropy and Gini/concentration
per-layer top-M expert coverage curves at feasible byte budgets
previous-token same-layer expert-overlap distribution
reuse-distance CDFs
session-conditioned vs global expert distribution divergence
static, per-layer LRU, session-frequency, and oracle cache curves
first cold-MB/token and PCIe-stall estimate
proceed / conditional / pivot memo
```

A frequency histogram alone is insufficient. It can show skew but cannot show whether the same experts recur soon enough, fit in the measured cache budget, or arrive before their deadlines.

### 23C-FAST.3 Decision rule

Classify the result rather than forcing a binary story:

```
PROCEED:
  oracle cache policy predicts a material stall reduction at the measured cache budget,
  and at least one causal heuristic captures a useful fraction of that gap on two or more domains.

CONDITIONAL:
  oracle headroom is strong but simple heuristics are weak or domain-specific;
  continue Observatory/simulator and test a lightweight predictor without promising runtime speedup.

PIVOT RUNTIME CLAIM:
  even the oracle provides little reduction in cold bytes or estimated stalls at feasible cache budgets;
  ship Observatory as the evidence-backed product and reassess the project direction immediately.
```

Planning thresholds for the memo, not universal scientific claims:

```
strong oracle signal: >=30% estimated blocking-stall reduction vs reactive
weak oracle signal:   <15% estimated blocking-stall reduction vs reactive
useful heuristic:     closes >=35% of the oracle-vs-LRU stall gap on held-out traces
```

Report sensitivity across cache sizes and PCIe bandwidth measurements instead of presenting one threshold as absolute truth.

---

## 23D. Predictor Training Plan

### 23D.1 Predictor is not the correctness core

**Hackathon scope rule:** Layer 1 begins with oracle, static-hotset, LRU/LFU, previous-token, and session-frequency baselines. A learned linear predictor is attempted only after these baselines and the byte-budget simulator are complete. Small MLPs, domain residual heads, and temporal predictors are post-MVP unless Layer 3 is already working.

The runtime must work with the predictor disabled:

```
true router + blocking fallback
session table
per-layer cache
optional global prior
```

The predictor is promoted only if it improves system metrics. This prevents a failed model-training effort from invalidating the whole project.

### 23D.2 Separate models

Train two distinct systems:

```
Domain classifier:
  estimates current domain mixture and topic shifts.

Expert activation predictor:
  ranks experts likely to be selected early enough for prefetch.
```

Do not conflate domain labeling with expert prediction.

### 23D.3 No-future-information leakage

For target layer L and token T, a training feature is allowed only if the production runtime would have it before the expert transfer deadline.

Allowed examples:

```
previous-layer router probabilities or top-M summary
previous-token same-layer expert IDs
session usage counts and decayed scores
global prior score
current domain mixture
position/context buckets
prefill/decode flag
cache-residency state
speculative offset after Layer 4
```

Forbidden examples:

```
target layer's router output used as an input to predict itself
future generated tokens
future-layer hidden states
features computed after the prefetch deadline
actual target expert IDs in the input
```

Every feature builder must declare:

```
source event order
target event order
availability offset
production computation cost
```

### 23D.4 Labels

Primary hard label:

```
128-way multi-hot vector with 1 for each selected routed expert
```

Soft labels:

```
full router probability distribution when available
or top-M router probabilities plus residual mass
```

Retain selected normalized weights separately from pre-top-k router probabilities. They represent different signals.

### 23D.5 Baselines before neural training

Evaluate in this order:

```
B0 reactive offload
B1 most frequent experts per layer
B2 previous-token same-layer experts
B3 session table only
B4 global prior only
B5 session + global prior
B6 session + prior + coactivation
P1 logistic/linear predictor
P2 one-hidden-layer MLP
P3 general predictor + domain residual head
P4 temporal model only if justified
```

A neural predictor must beat B6 after accounting for predictor latency and extra transfer bytes.

### 23D.6 Concrete first expert predictor

Inputs for target layer L:

```
previous layer top-M router probabilities
previous token same-layer 128-bit activation bitmap
short-window session frequency vector for layer L
long-window session frequency summary
domain mixture
layer embedding
position bucket
prefill/decode flag
cache-residency bitmap or compact summary
```

Model A — required first experiment:

```
per-layer logistic regression / linear 128-output classifier
```

Model B — only after Model A:

```
shared feature encoder
one hidden layer, 256 units
SiLU or ReLU
layer embedding
128-output layer-specific head
```

Do not begin with a transformer predictor.

### 23D.7 Domain-specific prediction

Do not train multiple full models initially. Use a shared general predictor plus optional small residual heads:

```text
final_logits = general_logits + domain_weight × domain_residual_logits
```

Activate at most one or two residual heads. Add a residual head only when:

```
it improves held-out downstream cache metrics in that domain
it does not regress mixed-domain sessions
it remains calibrated
its runtime cost is lower than the transfer time it saves
```

### 23D.8 Initial loss sequence

Layer 1:

```text
weighted binary cross-entropy over 128 expert membership labels
```

Layer 3, only if useful:

```text
BCE + soft-router distillation
```

Layer 4, only if simulator metrics improve:

```text
BCE + distillation + pairwise/listwise ranking term
```

Do not over-weight rare experts merely to maximize macro-F1. Aggressively predicting rare experts can create cache pollution.

### 23D.9 Scheduler separation

The predictor estimates probability of expert use. The scheduler decides whether the transfer is worthwhile.

```text
Predictor output:
  calibrated P(expert needed before deadline)

Scheduler value:
  probability
  × miss cost saved
  × reuse probability
  × urgency
  ÷ transfer bytes
```

Do not force the first predictor to learn PCIe cost, cache eviction, deadlines, and expert probability in one opaque objective.

### 23D.10 Calibration

After training:

```
fit temperature scaling
measure expected calibration error
measure Brier score
produce per-layer or grouped-layer reliability plots
```

Use a global temperature first. Adopt per-layer/grouped calibration only if validation data supports it.

### 23D.11 Uncertainty policy

Reduce predictor influence when:

```
output entropy is high
domain and general heads disagree
session table strongly disagrees
current feature distribution is out of training range
topic-shift detector fires
recent predictor usefulness falls
```

Fallback order:

```
session evidence
then global prior
then conservative high-confidence prefetch
then blocking exact load
```

### 23D.12 No online neural-weight updates in MVP

Runtime adaptation occurs through:

```
session table
recent-frequency windows
reuse-distance buckets
coactivation observations
domain recomputation
predictor-health weights
```

Neural weights remain frozen. New predictor versions are trained offline from versioned trace releases.

### 23D.13 Training and trace scale

**Two hundred conversations are enough for a pilot, not for a robust final predictor.** The pilot can establish locality, validate schemas, estimate cache curves, and compare non-neural policies. It can also train a disposable linear baseline, but its generalization claim must remain limited.

Scale by independent contexts and processed tokens:

```
Fast dry run:       24–40 conversations; directional locality decision only
Layer 1 pilot:      100–300 conversations; cache/simulator report and confidence intervals
Overnight expansion target after PASS:
                    >=250k processed tokens minimum
                    ~500k processed tokens target
                    1M processed-token cap unless learning curves still improve
                    aim for >=1,000 independent conversations when prompt lengths allow
Post-hackathon:     expand only from learning curves and rare-expert/domain coverage
```

With 30 routed layers, 250k processed tokens already produce roughly 7.5 million layer-token router events; 500k produce roughly 15 million. Those events are highly correlated within a conversation and must not be treated as millions of independent samples.

A 12–20 hour GPU collection run is allowed only after the fast locality gate passes. Before launching it:

```
freeze conversation-level train/validation/test membership
reserve at least one out-of-source test set
apply domain and length quotas
deduplicate source conversations and near-duplicate prompts
cap debug/full-probability trace storage
write append-only shards with resumable manifests
monitor thermals, disk, OOMs, malformed events, and collection throughput
```

More diverse data generally reduces overfitting. The failure modes are leakage, duplicated prompts, domain imbalance, and falsely treating correlated token/layer rows as independent evidence.

Track learning curves by:

```
independent conversations
processed prompt and decode tokens
source datasets and domains
rare-expert and expert-pair coverage
held-out source and leave-one-domain-out performance
calibration quality
simulated cold MB/token and stall time
performance vs trace volume
```

Stop the long run early if the latest shards add negligible held-out coverage or simulator improvement, or if collection instability threatens the submission schedule.

### 23D.14 Model selection metrics

Diagnostic metrics:

```
top-k expert recall
precision
macro/micro F1
ranking quality
calibration error
```

Promotion metrics:

```
cold MB/token
misses avoided per MB prefetched
misses avoided per millisecond of predictor latency
estimated PCIe stall time
cache pollution rate
p95 token latency in simulator
runtime throughput after Layer 3
```

Select checkpoints using the offline cache/PCIe simulator, not validation loss alone.

### 23D.15 Reproducibility and artifact contract

Every predictor artifact contains:

```rust
struct PredictorArtifactManifest {
    predictor_version: String,
    model_id: String,
    model_revision: String,
    quantization_profile: String,
    runtime_compatibility: String,
    trace_dataset_id: String,
    split_manifest_hash: String,
    feature_schema_hash: String,
    architecture: String,
    loss_version: String,
    calibration_version: String,
    training_seed: u64,
    training_metrics: Metrics,
    simulator_metrics: Metrics,
}
```

The runtime must reject incompatible artifacts rather than silently falling back to incorrect feature interpretation.

---

## 23E. Data and Training Quality Gates

### 23E.1 Collector integration gate

```
All Gemma4TextRouter modules discovered.
Tiny-random model emits expected top-k events.
Prefill and decode token mapping verified manually.
Hooked and unhooked generation produce identical tokens/logits within tolerance.
Malformed output shapes fail loudly.
Manifest and file hashes are produced.
```

### 23E.2 Pilot real-model gate

```
At least 100 real Gemma conversations traced.
No missing layer/token events.
Deterministic replay parity confirmed on a sample.
Core Parquet conversion works.
Basic frequency, reuse-distance, and coactivation plots look internally consistent.
```

### 23E.3 Dataset gate

```
Conversation-level splits frozen.
No duplicate conversation across train/validation/test.
Domain and length coverage meet declared quotas.
At least one out-of-source test set exists.
Licenses and privacy handling are documented.
Trace manifest binds data to model/runtime profile.
```

### 23E.4 Predictor baseline gate

```
B0–B6 baselines reproduced.
Oracle predictor upper bound measured.
Linear predictor evaluated at fixed byte/prefetch budgets.
Calibration measured.
No hidden future-information leakage found in feature audit.
```

### 23E.5 Neural promotion gate

A predictor may enter Layers 3/4 runtime experiments only if:

```
it beats session + prior + coactivation on held-out simulator traces
it reduces cold MB/token or estimated stall time materially
its inference overhead is below the saved transfer time
it does not materially regress topic-shift or out-of-source tests
its probabilities are sufficiently calibrated for expected-value scheduling
```

---

## 23F. Immediate layered implementation sequence

The sequence protects next-day evidence, the Observatory contingency, and a full video-only day.

```
LAYER 0 — 2026-07-14
1. Create the repository, locked environment, experiment ledger, and Codex task log.
2. Run hardware/CUDA/model-access diagnostics.
3. Validate router hooks and parity on tiny-random/gemma-4-moe.
4. Attempt the pinned 4-bit target load and record the real memory profile.
5. Run the 24–40 conversation stratified locality probe.
6. Generate concentration, reuse, overlap, working-set, and first cache curves.
7. Publish PROCEED / CONDITIONAL / OBSERVATORY-ONLY memo.
8. Start the balanced overnight trace run only after a positive or conditional gate.

LAYER 1 — COMPLETE BY 2026-07-15, FREEZE 2026-07-16
9. Validate overnight data and freeze conversation-level manifests.
10. Complete measured expert-size and PCIe microbenchmarks.
11. Complete reactive/static/LRU/session/oracle event-driven simulations.
12. Publish the evidence bundle and runtime go/no-go decision.
13. Generate the minimum static replay report while the GPU continues collecting.

LAYER 2 — COMPLETE AND FREEZE BY 2026-07-17
14. Generate a machine-specific recommendation from the same simulator artifacts.
15. Add measured VRAM allocation/headroom reporting; do not add KV experiments.
16. Make one causal timeline understandable without MoE expertise.
17. Bundle the tiny-model smoke test, trace replay, manifests, and one-command evaluator.
18. Rehearse and screen-test the complete Observatory-only demo.

LAYER 3 — ATTEMPT 2026-07-16 TO 2026-07-18 ONLY
19. Implement the thin Gemma adapter and exact blocking reactive baseline.
20. Add preallocated expert slots, per-layer LRU/session residency, async copy, and telemetry.
21. Stop immediately if exactness, slot execution, or measurable baseline comparison remains unresolved at cutoff.
22. Add no neural predictor unless the simulator proves a material heuristic gap and the runtime is already stable.

FREEZE AND MEDIA — 2026-07-19 TO 2026-07-21
23. On July 19, freeze code, prompts, manifests, claims, scorecards, README, and both scripts.
24. Select Demo A only when Layer 3 passes; otherwise select Demo B without apology or ambiguity.
25. Use July 20 exclusively for recording, editing, uploading, processing, and link verification.
26. Use July 21 only for reproduction checks, form completion, and emergency non-architectural corrections.
```

Do not wait for the overnight dataset to implement schemas, simulations, reports, or replay. Do not let Layer 3 delay the Layer 2 freeze.

---

## 24. Offline Simulator

The Layer 1 simulator is an event-driven timeline model, not a cache-hit spreadsheet. It consumes router traces plus measured hardware microbenchmarks and predicts when each expert becomes ready relative to its compute deadline.

### 24.1 Inputs

```
actual per-token/per-layer selected experts
expert object bytes by layer and runtime profile
measured pageable->pinned and pinned->GPU transfer latency curves
measured fixed transfer/launch overhead
measured per-layer compute durations
GPU expert-slot capacity by layer
pinned-memory budget
copy-stream count and queue discipline
prefetch issue time and expert deadline
optional lookahead/MTP window
```

Use measured effective bandwidth from multiple transfer sizes. Do not model every DMA as theoretical PCIe bandwidth.

### 24.2 Policies to compare

```
B0 reactive blocking offload, no retained expert cache
B1 static per-layer frequency placement
B2 global LRU
B3 per-layer LRU
B4 per-layer LRU + decayed LFU/admission
B5 session-frequency cache, no prefetch
B6 session cache + heuristic prefetch
B7 global prior + session cache
B8 realistic linear predictor at a fixed byte budget
B9 oracle prefetch upper bound
B10 Layer 4 lookahead/MTP controller
```

The oracle is an upper bound, never a product baseline.

### 24.3 Required simulated events

For every expert demand and movement, emit:

```
request_id, token_index, layer_id, expert_id
prediction_source
state_before, state_after
router_needed_at_us
prefetch_queued_at_us
copy_start_us, copy_end_us
expert_compute_start_us
bytes_transferred
slot_id
was_eventually_used
eviction_time_us and eviction_reason
```

### 24.4 Primary simulator metrics

```
ready_hit_rate
late_prefetch_rate
blocking_miss_rate
cold_MB_per_token
useful_prefetch_MB_per_token
wasted_prefetch_MB_per_token
transfer_amplification
estimated_PCIe_stall_ms_per_token
copy_compute_overlap_percent
prefetch_deadline_miss_rate
deadline_slack_us distribution
cache churn and repeated reload count
eviction regret
```

Ordinary cache hit rate is secondary: an expert is only a **ready hit** when it is resident before its compute deadline.

### 24.5 Calibration

After Layer 3 exists, replay the same trace through the simulator and runtime. Report simulator error for:

```
transfer duration
blocking stall duration
ready-hit rate
cold MB/token
p50/p95 token latency
```

The simulator is accepted only as a planning tool; measured runtime results remain authoritative.

### 24.6 Layer 1 go/no-go gate

Proceed to live runtime optimization when both are true:

```
1. Oracle policy predicts >= 1.5x throughput-equivalent improvement over reactive offload
   or >= 50% lower PCIe stall time on at least one declared workload.

2. A realistic non-oracle policy beats per-layer LRU by >= 20% cold MB/token
   or >= 25% estimated PCIe stall time on held-out traces.
```

If the oracle is weak, stop runtime work and submit the Observatory as an evidence-backed negative result and configuration profiler.

---

## 25. MVP Runtime Policy

Layer 3 is the live, exact runtime **without MTP or speculative decoding**.

### 25.1 Locked scope

```
model: Gemma 4 26B A4B, text-only
batch size: 1
host runtime: pinned Hugging Face Transformers/PyTorch revision
runtime profile: one Stage-1-selected profile used by every baseline
backbone/router/state: host runtime implementation
routed experts: ExpertFlow adapter
cache unit: one expert in one layer
copy: pinned host memory + one asynchronous CUDA copy stream initially
cache: per-layer GPU slot arena
policy: session frequency + decayed recency + admission threshold
predictor: disabled initially; simple heuristics first
fallback: exact blocking transfer
speculation: disabled
```

### 25.2 Minimum live baselines

All baselines execute in the same adapter and differ only in policy:

```
R0 reactive: evict after use; no prefetch
R1 static: fixed top experts per layer
R2 per-layer LRU: retain on demand; no prediction
R3 ExpertFlow: session-aware admission + asynchronous heuristic prefetch
R4 ExpertFlow + linear predictor: optional
```

This avoids attributing unrelated attention, tokenizer, quantization, or kernel differences to ExpertFlow.

### 25.3 Initial non-neural prefetch signals

```
previous-token same-layer expert reuse
current-session per-layer frequency and recency
prefill hot set
previous-layer/current-token coactivation hints when available before deadline
static global per-layer prior
victim-cache bounce-back
```

A linear model is added only if it improves fixed-byte-budget simulator and runtime metrics.

### 25.4 Exactness

```
The true router always executes.
Every selected expert is executed.
A miss blocks until the exact expert is ready.
Prediction never changes model output.
```

### 25.5 Runtime telemetry is part of the product

The runtime must emit the same event schema as the simulator, using CUDA events for copy and compute timestamps. A speed number without movement telemetry does not satisfy Layer 3.

---

## 26. Speculative/Draft Roadmap

Layer 4 adds optional lookahead only after the exact no-MTP runtime is demonstrably useful.

### 26.1 Purpose

Lookahead is used to issue expert transfers earlier. It is not allowed to change verifier correctness.

### 26.2 Supported Layer 4 shape

Implement one lookahead provider, whichever is reliable in the chosen host runtime:

```
external small draft model
or native/model-provided MTP exposed by the pinned runtime
```

Do not build both during the hackathon.

### 26.3 Rollout

```
1. window = 1, telemetry only
2. window = 1, speculative prefetch enabled
3. dynamic window 1–2
4. larger window only if cold expert union remains within byte budget
```

### 26.4 Controller

```
if acceptance high and predicted cold expert union small:
    allow lookahead window 2
elif queue depth, late-prefetch rate, or cache pollution rises:
    shrink to window 1
else if Layer 3 is faster:
    disable lookahead
```

Control by bytes and deadlines, not token count alone.

### 26.5 Layer 4 success rule

MTP/lookahead is enabled in the final demo only if it provides:

```
>= 10% decode TPS improvement over Layer 3
and no >5% regression in p95 inter-token latency
and no exactness regression
and no material increase in wasted prefetch MB/token
```

A controller that correctly disables harmful speculation is a valid engineering result.

### 26.6 Qwen

Qwen3.5 native MTP and DeltaNet state handling are post-hackathon work. They are not a fallback if Gemma Layer 3 is unfinished.

---

## 27. Qwen / DeltaNet Risk Section

Qwen3.5-35B-A3B interleaves Gated DeltaNet and Gated Attention blocks with MoE.

This matters because:

```
Gated DeltaNet uses recurrent-style state, not a standard KV cache.
State size and reuse behavior differ from normal attention KV cache.
MTP verification may require careful recurrent-state management.
Prefix reuse and rollback may need snapshots.
Backbone residency cannot be summarized as "attention parts in VRAM."
```

Before Qwen integration, produce a dedicated memory/state report:

```
Gated DeltaNet state shape
Gated Attention KV shape
state growth with sequence length
state snapshot cost
rollback cost for speculation
non-expert VRAM footprint
expert-cache space remaining
```

Qwen remains post-hackathon Stage 5+ until this report exists.

---

## 28. Metrics

TPS is necessary for the pitch but insufficient for engineering evaluation. The evaluation must prove three outcomes: **Capability, Performance, and Mechanism**.

### 28.1 North-star scorecards

#### Capability

```
model and exact runtime profile
full model/host-memory footprint
peak GPU VRAM and peak host RAM
available routed-expert cache slots
measured KV/model-state allocation at the tested context
remaining configurable VRAM headroom
successful completion under the declared hardware envelope
exactness result
```

#### Performance

```
decode tokens/sec
p50/p95/p99 inter-token latency
time to first token
prefill tokens/sec
blocking transfer ms/token
same-runtime reactive and per-layer LRU comparisons
```

#### Mechanism

```
ready-prefetch rate
late-prefetch rate
blocking-miss rate
cold MB/token
useful vs wasted prefetch MB/token
copy/compute overlap
transfer amplification
output parity and identical evaluation envelope
```

The main claim is Layers 3/4 versus same-runtime reactive offload. A result is not credited to ExpertFlow if it comes from different prompts, output lengths, quantization, kernels, cache budgets, or hardware settings.

Recommended submission scorecard:

| Result | Reactive baseline | ExpertFlow |
|---|---:|---:|
| Tokens/second | measured | measured |
| p95 inter-token latency | measured | measured |
| Blocking transfer time | measured | measured |
| Ready expert hits | measured | measured |
| PCIe transfer amplification | measured | measured |
| Output parity | pass/fail | pass/fail |
| Peak VRAM | measured | measured |
| Remaining VRAM headroom | measured | measured |

### 28.2 Expert demand classification

Every selected expert is assigned exactly one outcome:

```
HOT_HIT           resident before it was predicted or demanded
READY_PREFETCH     copied because of a prediction and ready before compute
LATE_PREFETCH      copy started early but completed after its deadline
BLOCKING_MISS      no usable early copy; token/layer waited
VICTIM_HIT         recovered from victim cache without host transfer
CPU_COMPUTE        only if an explicit CPU-execution baseline exists
```

Every prefetch that is never used before eviction is marked `WASTED_PREFETCH`.

### 28.3 Event-level movement schema

```rust
struct ExpertMovementEvent {
    request_id: String,
    token_index: u64,
    layer_id: u16,
    expert_id: u16,
    prediction_source: PredictionSource,
    state_before: ResidentState,
    state_after: ResidentState,
    router_needed_at_us: Option<u64>,
    prefetch_queued_at_us: Option<u64>,
    copy_start_us: Option<u64>,
    copy_end_us: Option<u64>,
    expert_compute_start_us: Option<u64>,
    eviction_at_us: Option<u64>,
    bytes: u64,
    slot_id: Option<u32>,
    outcome: ExpertDemandOutcome,
    eviction_reason: Option<EvictionReason>,
}
```

Derived deadline slack:

```
deadline_slack_us = expert_compute_start_us - copy_end_us
positive: ready early
near zero: just in time
negative: late and contributed to a stall
```

### 28.4 PCIe and overlap metrics

```
PCIe bytes/token
cold MB/token
useful prefetch MB/token
wasted prefetch MB/token
transfer amplification = all transferred bytes / minimum required cold bytes
copy stream busy percentage
copy/compute overlap percentage
blocking PCIe stall ms/token
prefetch queue p50/p95 depth
copy launch count/token
batched transfer size distribution
```

### 28.5 Cache metrics

```
ready-hit rate, overall and per layer
ordinary resident hit rate
late-prefetch rate
blocking-miss rate
victim-hit rate
cache pollution rate
expert reload count
cache churn MB/token
eviction regret: evicted expert reused within N tokens
reuse-distance capture rate
working-set coverage by layer
```

### 28.6 Predictor metrics

Evaluate predictors at equal prefetch-byte budgets:

```
expert recall before deadline
useful prefetch precision
misses avoided per MB prefetched
stall milliseconds avoided per MB prefetched
misses avoided per millisecond of predictor overhead
calibration error
false-prefetch MB/token
per-source usefulness
```

Raw top-k accuracy alone is not a promotion metric.

### 28.7 Speculative metrics

```
draft acceptance rate
lookahead/MTP window
unique expert union per verifier pass
cold expert union bytes per verifier pass
partial-prefix acceptance
speculation-induced cache pollution
Layer 4 incremental TPS and p95 change versus Layer 3
```

### 28.8 Exactness metrics

For fixed deterministic cases:

```
same model/runtime profile
same prompt and chat template
same seed and sampling settings
same selected experts
same generated tokens
logits within declared numerical tolerance
```

### 28.9 Evaluation suite

Freeze a manifest-driven suite with:

```
workloads: general chat, code, math, translation, multilingual/code-switch,
           structured JSON, stable long session, abrupt topic shift
context buckets: ~512, ~2K, optional ~8K tokens
generation: 64 and 128 new tokens
cache states: cold start and warm continuation
sampling: deterministic for parity; one controlled stochastic robustness subset
repetitions: one warm-up plus at least five measured runs when time permits
```

Report medians and p95, not the single fastest run.

### 28.10 Evaluation artifacts

Every run writes:

```
run_manifest.json
summary.json
expert_movements.jsonl
per_token_latency.jsonl
hardware_and_versions.json
optional Chrome trace / Perfetto timeline
```

The dashboard must be able to replay these files without loading the model.

---

### 28.11 Causal dashboard acceptance test

A person unfamiliar with MoE inference should be able to answer the following after watching one replay:

```
Which expert was predicted?
When did its transfer start and finish?
When did the real router request it?
Was it ready, late, or completely missing?
How many milliseconds of blocking time were avoided or incurred?
Did the output remain exact?
```

The dashboard fails the product test if these answers require reading raw logs or understanding implementation-specific class names.

---

## 29. Success Criteria

### 29.1 Layer 0 — empirical gate

```
pinned 4-bit profile load succeeds or fails with a reproducible diagnostic
router hooks pass parity and integrity checks
dry-run traces cover the declared domains
expert sizes and first transfer measurements are available
PROCEED / CONDITIONAL / OBSERVATORY-ONLY decision is written by 2026-07-15
```

### 29.2 Layer 1 — Observatory core, required

```
real-model trace pilot is valid and reproducible
reactive, static, LRU, session, and oracle policies run from one trace schema
ready/late/blocking events and PCIe deadlines are represented
locality and working-set conclusions use held-out conversations where applicable
all estimates are tied to a hardware/model/runtime manifest
```

### 29.3 Layer 2 — recommendation and replay product, required

```
machine-specific recommendation is generated from measured/simulated artifacts
VRAM allocation and remaining headroom are reported without unmeasured KV claims
one causal timeline explains predicted, transferred, ready, late, and blocking events
bundled replay works without the full model or matching GPU
Demo B can be recorded end-to-end by 2026-07-18
```

Layers 1 and 2 define a complete Observatory submission.

### 29.4 Layer 3 — exact live runtime, target

Required correctness:

```
selected-expert and generated-token parity on deterministic tests
forced misses complete correctly
no use-after-evict or prefetch/eviction race
bounded pinned-memory and GPU-slot use
```

Required observability:

```
every demand classified
copy and compute timestamps recorded
bytes and slot transitions reconciled
runtime trace replayable through the same report path
```

Performance tiers versus the same-runtime reactive baseline:

```
minimum useful result: >= 1.20x decode TPS or >= 20% lower p95 latency
competitive target: >= 1.50x decode TPS
stretch: >= 2.00x decode TPS
```

The same-runtime per-layer LRU baseline must be reported. If ExpertFlow does not improve cold MB/token, stall time, p95 latency, or throughput beyond LRU, frame the result as an exact dynamic offload implementation rather than a predictive-cache win.

### 29.5 Layer 4 — adaptive lookahead/MTP, optional

```
Layer 3 already passed and is frozen
exact verifier parity preserved
incremental benefit measured against Layer 3
p95, cold-union, and wasted-prefetch budgets respected
automatic disable path works
```

### 29.6 Submission selection gate

Choose **Demo A** only when Layer 3 is exact, reproducible, and has a defensible same-runtime result by the July 18 cutoff.

Choose **Demo B** when any of the following is true:

```
Layer 3 remains incomplete or unstable
measured performance does not beat the relevant baseline
runtime integration would consume July 19 or video day
runtime evidence is weaker than the Observatory evidence
```

No last-minute hybrid story is allowed. The selected script, scorecards, README opening, and submission description must all describe the same outcome.

### 29.7 Winning proof hierarchy

```
OBSERVATORY PROOF — required
  real hardware and router measurements
  model-specific locality verdict
  event-driven policy comparison
  machine-specific recommendation
  causal replay and judge reproduction

RUNTIME PROOF — target
  exact live execution in the declared envelope
  material same-runtime improvement
  more experts ready before deadlines
  remaining VRAM headroom reported under the same profile

TRUST — required in both outcomes
  measured and estimated results visibly separated
  unsupported claims and future-model implications labeled
  exactness claimed only with parity artifacts
```

### 29.8 Claim-discipline gate

```
No universal "runs any MoE" claim.
No statement that existing offloaders are mostly disk streaming.
No simulated number presented as measured.
No future workstation-scale model presented as already demonstrated.
No conversion from free VRAM megabytes to context tokens or concurrent agents without measurement.
No speedup claim without same-runtime reactive and LRU baselines.
No "exact" claim without parity artifacts.
```

---

## 30. Configuration

```yaml
project:
  track: developer_tools
  required_layers: [layer_1_observatory, layer_2_recommendation_replay]
  target_layer: layer_3_exact_runtime
  stretch_layer: layer_4_adaptive_lookahead
  official_demo_selection_deadline: 2026-07-18
  all_product_freeze: 2026-07-19
  video_only_day: 2026-07-20

model:
  primary_poc: google/gemma-4-26b-a4b-it
  post_hackathon_target: Qwen/Qwen3.5-35B-A3B
  modality: text_only
  batch_size: 1
  runtime_profile_sequence: [int4_weight_only, int8_weight_only_after_int4_gate]
  hackathon_reference_profile: int4_weight_only
  int4_backend_candidate: torchao_stable
  int4_backend_and_layout: pinned_single_profile_after_slot_swap_gate
  repack_on_cache_miss: false
  silent_precision_fallback: false
  do_not_mix_profiles_between_baselines: true

runtime:
  host: huggingface_transformers_pytorch
  revisions_pinned: true
  llama_cpp_critical_path: false
  build_new_full_inference_engine: false
  replace_only_routed_expert_path: true
  initial_copy_streams: 1
  use_cuda_events: true
  preallocate_gpu_expert_slots: true
  public_cli_only_for_demo: true
  environment_lock: uv_lock_and_pinned_cuda_container
  doctor_required_before_live_run: true

hardware:
  gpu_vram_gb: 16
  cpu_ram_preferred_gb: 64
  pcie_bandwidth_source: measured_microbenchmark
  cpu_warm_pinned_budget: bounded_and_measured

cache:
  unit: layer_expert
  per_layer_partitioning: true
  replacement_policy_layer_3: lru_plus_decayed_lfu
  admission_policy_layer_3: frequency_recency_threshold
  victim_cache_sizing: trace_derived
  prefetch_buffer_fraction: simulator_selected
  exact_blocking_fallback: true

predictor:
  layer_1: oracle_and_heuristic_baselines
  layer_3_initial: previous_token_plus_session_frequency
  optional: linear_router_feature_predictor
  mlp_and_temporal_models: post_mvp_unless_layer_3_complete

speculation:
  layer_3: disabled
  layer_4_provider: choose_one_external_draft_or_native_mtp
  layer_4_window: dynamic_1_to_2
  control_metric: predicted_cold_expert_union_bytes
  disable_when_harmful: true

metrics:
  headline: [decode_tps, p95_inter_token_latency, peak_vram, exactness]
  primary_internal: [ready_hit_rate, blocking_stall_ms_per_token, cold_mb_per_token]
  event_trace_required: true
  simulator_runtime_calibration_required: true

reproducibility:
  modes: [tiny_model_smoke, bundled_trace_replay, full_live_4bit]
  bundled_trace_without_private_text: true
  checked_in_expected_result_manifest: true
  cpu_ci_required: true
  gpu_ci_optional_but_recommended: true
  model_revision_and_chat_template_hash_required: true

eval:
  same_runtime_baselines: true
  deterministic_parity_suite: true
  measured_repetitions_target: 5
  report_median_and_p95: true

trace_collection:
  fast_dry_run_conversations: 24_to_40
  layer_1_pilot_conversations: 100_to_300
  overnight_run_only_after_locality_pass: true
  overnight_processed_tokens_minimum: 250000
  overnight_processed_tokens_target: 500000
  overnight_processed_tokens_cap: 1000000
  splits_frozen_before_long_run: true
  deduplication_and_domain_quotas_required: true
  learning_curve_stop_rule: true

adaptive_learning_post_hackathon:
  change_true_router: false
  first_online_method: constrained_contextual_bandit
  full_rl: only_if_bandit_insufficient
  shadow_mode_required: true
  deterministic_fallback_required: true
```

---

## 31. Layered Delivery Roadmap

The product is built from independently shippable layers. Later layers may improve the submission but may never invalidate or delay earlier ones.

### Layer 0 — Empirical gate

```
4-bit load and slot feasibility
router-hook validation
stratified locality dry run
expert-size and PCIe measurements
written runtime decision
```

### Layer 1 — Observatory core, must ship

```
trace collector and validator
locality/reuse/working-set report
event-driven simulator
reactive/static/LRU/session/oracle policies
ready/late/blocking metrics
```

### Layer 2 — Recommendation and replay, must ship

```
shared analysis artifact pipeline
machine-specific configuration recommendation
measured VRAM allocation and headroom
single causal timeline
static/lightweight interactive report
bundled judge replay and expected manifest
complete Observatory-only demo assets
```

Layer 2 is deliberately smaller than the former all-at-once product-polish list. The following are **not required** for the submission floor:

```
full live multi-panel dashboard
separate frontend for every CLI command
KV-cache benchmark
session-concurrency scheduler
neural predictor
MTP controller
multiple installation methods beyond one locked primary path plus replay
```

### Layer 3 — Exact live runtime, target

```
Hugging Face Gemma adapter
pinned 4-bit expert objects
CPU store and bounded pinned staging
preallocated GPU slots
reactive and LRU baselines
session-aware residency and asynchronous prefetch
exact fallback and runtime telemetry
same-runtime verification
```

### Layer 4 — Adaptive lookahead/MTP, stretch

```
one lookahead provider
window-1 telemetry
cold-union-aware transfer decisions
window 1–2 only if beneficial
automatic disable path
```

### Post-hackathon layers

```
adaptive cache-policy learning and contextual bandits
Qwen/DeltaNet feasibility and runtime work
llama.cpp/GGML and other backend adapters
fused kernels, continuous batching, multi-user scheduling
long-context, multimodal, and multi-GPU work
```

### Stop rules

```
Do not start Layer 3 when the Layer 1 oracle is weak.
Do not allow Layer 3 to delay the complete Layer 2 contingency product.
Stop Layer 3 by the July 18 cutoff if exactness and comparison are unresolved.
Do not start Layer 4 unless Layer 3 is already exact, faster, and video-ready.
Do not train a more complex predictor while telemetry or baselines are incomplete.
Do not add new product or technical scope after July 19.
Do not write or debug features on July 20 video day.
```

---

## 32. Implementation Risks

### 32.1 Insufficient expert locality
Mitigation: Layer 1 trace analysis before runtime work, strict admission control, and per-domain priors only if they beat simpler baselines.

### 32.2 PCIe bottleneck
Mitigation: pipeline simulator, async DMA, pinned memory, batch transfers, reduce speculative expert union.

### 32.3 Non-expert components consume too much VRAM
Mitigation: Gemma VRAM breakdown first, short-to-medium context initially, quantize non-expert components if acceptable.

### 32.4 Predictor cache pollution
Mitigation: measure useful_prefetch_rate, cap prefetch buffer, penalize bad predictors by source, require admission threshold.

### 32.5 Topic shifts
Mitigation: domain classifier, JS-divergence shift detector, α floor, faster session decay on shift, victim cache.

### 32.6 Speculation increases expert diversity
Mitigation: cold-expert-union budget, partial prefix acceptance, dynamic window control, and speculation only in Layer 4.

### 32.7 Qwen architecture complexity
Mitigation: Gemma first; Qwen memory/state report and runtime work only after the hackathon Layer 4 path is complete.


### 32.8 Technical contribution is difficult to understand
Mitigation: make the causal timeline the primary visualization; explain one expert's predicted-transfer-demand lifecycle before showing aggregate charts; keep the three-minute demo focused on product evidence rather than architecture inventory.

### 32.9 Product stops at analysis and never closes the loop
Mitigation: `expertflow recommend` must emit a runnable configuration and explanation; `expertflow verify` must compare that run with fixed baselines. Charts alone do not satisfy the product goal.

### 32.10 Future impact is overstated
Mitigation: separate proven hackathon results, immediate product value, and future sparse-model implications in every public description. Do not name a future model as supported unless it has been tested. Report freed or reserved VRAM as headroom; do not claim a particular context-length or concurrency gain without measuring it.

---

## 33. Open Questions

The implementation-framework question is closed for the hackathon: use Hugging Face/PyTorch plus a thin custom expert backend, not llama.cpp and not a new full runtime.

Immediate measurable questions:

1. What is the routed expert object size under each viable runtime profile?
2. How should VRAM be divided between preallocated expert slots, KV/model state, transfer buffers, scratch space, and safety reserve for the target local-agent workload?
3. What transfer-size curve is achieved from bounded pinned host memory on the target machine?
4. What percentage of resident hits are actually ready before the expert deadline?
5. How much locality comes from previous-token reuse, session frequency, and prefill hot sets?
6. Does a realistic heuristic policy beat per-layer LRU under the same byte budget?
7. How well does the event-driven simulator predict measured Layer 3 stalls?
8. Can the PyTorch expert adapter execute the chosen quantized profile without dequantization/copy overhead erasing the benefit?
9. Does a linear predictor improve stall time after its own latency and false-prefetch bytes are included?
10. Does Layer 4 lookahead reduce late prefetches, or does its cold expert union increase cache pollution?
11. At equal exactness and runtime settings, what expert-cache budget gives the best Pareto trade-off between ready hits and remaining agent-state headroom?

Post-hackathon questions:

12. Is Qwen3.5's DeltaNet state manageable for exact MTP verification and rollback?
13. Can the backend interface be ported cleanly to llama.cpp/GGML or an existing heterogeneous MoE runtime?
14. Does a contextual bandit improve admission/prefetch decisions over the frozen heuristic without increasing p95 latency or transfer amplification?
15. Which reward horizon is necessary to capture eviction regret and future cache pollution without requiring full RL?

16. What minimum profile duration produces a stable enough recommendation for a new machine/model pair?
17. How should recommendation confidence be calibrated when the oracle has headroom but learned or heuristic policies do not?
18. Which dashboard view most clearly communicates a ready prefetch versus a late prefetch to a non-specialist?
19. What evidence is sufficient to recommend disabling predictive caching for a profile?

---

## 34. Final Product Thesis

ExpertFlow Local is a developer tool for turning sparse routed weights into a **measured, simulated, and optimized memory hierarchy** across GPU VRAM and host RAM, so local agent runtimes can balance expert readiness against active model-state headroom.

The product thesis is:

> **A sparse model's total parameter count should not automatically determine its minimum GPU tier—or consume the entire VRAM budget of a local agent. The relevant systems question is whether its active routed working set can be kept ready before compute deadlines while preserving useful headroom for model state and sessions.**

The product loop is:

```
Inspect the machine.
Observe the real router.
Measure the working set.
Simulate cache and PCIe deadlines.
Recommend a machine-specific allocation between routed experts and remaining runtime headroom.
Run the exact model.
Visualize every movement.
Verify the improvement causally.
```

The essential invariants remain:

```
Prediction decides what to load early.
The router decides what is true.
Measured runtime results outrank simulated estimates.
A recommendation may legitimately be "do not enable predictive caching."
```

The value appears at three levels:

1. **Profiler and feasibility tool:** developers learn whether a model's routed working set is compatible with their machine.
2. **Policy laboratory and autotuner:** developers compare reactive, static, LRU, session-aware, predictor, and oracle policies and receive a runnable configuration that states both expected expert readiness and remaining VRAM headroom.
3. **Exact runtime:** the cache backend reduces or hides host-to-device stalls without changing the selected experts or generated output.

The hackathon claim must remain narrow and provable:

> **On the tested Gemma 4 Q4 profile and 16 GB GPU target, ExpertFlow measures expert locality, transfer deadlines, and the VRAM trade-off between routed-expert residency and remaining runtime headroom; if the live gate passes, it reduces measured blocking transfer time versus a same-runtime reactive baseline while preserving exact behavior.**

The long-term implication is broader but explicitly unproven:

> **Predictive working-set management could lower the hardware tier required by future large-total-parameter, low-active-parameter models while leaving more of a workstation's VRAM available to the active context and concurrent sessions of private local agents.**

A polished Observatory is a complete developer product. A successful Layer 3 runtime is the technical proof that elevates it. Layer 4 MTP is an optional multiplier, not the foundation.

---

## 35. External Sources and Implementation References

Primary technical and product-framing references reviewed for v0.5–v0.9:

1. Hugging Face Transformers Gemma 4 documentation  
   https://huggingface.co/docs/transformers/model_doc/gemma4

2. Hugging Face Transformers Gemma 4 implementation (`Gemma4TextRouter`)  
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py

3. MoE-Beyond paper  
   https://arxiv.org/abs/2508.17137

4. MoE-Beyond repository, collector scripts, example traces, and training data  
   https://github.com/ngavhane/moe-beyond

5. MoE-Infinity paper  
   https://arxiv.org/abs/2401.14361

6. MoE-Infinity repository  
   https://github.com/EfficientMoE/MoE-Infinity

7. Public MoE Expert Selection Trace dataset  
   https://huggingface.co/datasets/core12345/MoE_expert_selection_trace

8. Tiny random Gemma 4 MoE integration-test checkpoint  
   https://huggingface.co/tiny-random/gemma-4-moe

Candidate public conversation sources:

9. Pure-Dove  
   https://huggingface.co/datasets/LDJnr/Pure-Dove

10. OpenAssistant OASST1  
    https://huggingface.co/datasets/OpenAssistant/oasst1

11. UltraChat 200k  
    https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

12. WildChat  
    https://huggingface.co/datasets/allenai/WildChat

13. Capybara  
    https://huggingface.co/datasets/LDJnr/Capybara


14. llama.cpp repository — quantization and CPU+GPU hybrid inference reference; not the hackathon critical path  
    https://github.com/ggml-org/llama.cpp

15. KTransformers repository — heterogeneous expert placement and CPU/GPU scheduling reference  
    https://github.com/kvcache-ai/ktransformers

16. PyTorch CUDA semantics — streams and events for asynchronous transfer instrumentation  
    https://pytorch.org/docs/stable/notes/cuda.html

17. NVIDIA CUDA C Programming Guide — asynchronous copies, streams, and event timing  
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/

18. TorchAO repository and INT4 weight-only configuration — first 4-bit backend candidate  
    https://github.com/pytorch/ao

The source list is not a blanket redistribution permission. Verify each artifact's current license, access conditions, and dataset terms before use or publication.


Product and submission references added in v0.9:

- OpenAI Build Week rules and judging criteria  
  https://openai.devpost.com/rules

- NVIDIA RTX PRO 6000 Blackwell product family, used only as a future workstation-class illustration; not a tested ExpertFlow target  
  https://www.nvidia.com/en-us/design-visualization/rtx-pro-6000-blackwell-workstation-edition/
