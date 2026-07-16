# Generic Multi-Layer C5 Cache Design

**Status:** Approved for test-first implementation after diagnostic benchmark commit `c72f578`. No llama.cpp source change is part of this design milestone.

## Goal and limits

Generalize the exact C5 primitive from one layer to an explicit set of Gemma MoE layers, retaining exactly 32 direct packed CUDA slots per enabled layer. The cache remains reactive, blocking, true-router driven, and disabled by default. It adds no prediction, asynchronous copy stream, MTP/ML, 64-slot allocation, kernel change, graph rewrite, or per-token allocation.

The source model and canonical traces contain 30 MoE layers, numbered `0..29`, each routing eight experts. Expansion is gated in this order:

1. two layers: `[0,24]`
2. five layers: `[0,7,14,21,29]`
3. all intended layers: `[0,1,...,29]`

Layer 24 retains the proven C5 boundary. The five-layer set is evenly spread through the model and spans train+validation decode LRU-32 hit rates from 57.80% to 80.00% (`0=57.80`, `7=68.22`, `14=80.00`, `21=74.65`, `29=60.26`). No expanded-test trace was consulted to select these layers.

## Architecture

### Per-layer contexts

Replace singleton scheduler state with a fixed-capacity array indexed by physical layer ID. Each enabled entry owns:

- one independent 32-slot LRU state and access sequence;
- exact source and scheduler-copy bindings for gate/up weight, down weight, and down scale;
- three arena tensor views and their aligned offsets;
- fixed-capacity event records and per-layer counters;
- enabled, initialized, and validation state.

Layer contexts never share residency, generations, recency, or logical-to-physical mappings. A selected-ID tensor is prepared against the context identified by the parsed `blk.N` tensor name. The authoritative top-8 order and weights remain unchanged; only logical expert IDs are replaced by verified physical slot IDs immediately before the unchanged `MUL_MAT_ID` operations.

### Generic tensor identification

Identify only exact names:

- `blk.N.ffn_gate_up_exps.weight`
- `blk.N.ffn_down_exps.weight`
- `blk.N.ffn_down_exps.scale`

Parse `N` as a decimal integer in `0..29`, require that it is explicitly enabled, and validate the exact Gemma Q4_0/F32 type, shape, stride, packed byte count, and 128-expert cardinality already proven at layer 24. Reject partial names, leading/trailing text, unsupported layers, duplicate components, and metadata mismatches.

### Consolidated CUDA arena

Allocate one scheduler-owned CUDA backend buffer once, after all enabled-layer bindings are complete. Within it, assign each layer three aligned regions in ascending configured-layer order: gate/up, down, scale. Every region uses the backend-reported allocation size and alignment; the final buffer size is padded to that same alignment.

The projected packed payload is `107,053,184` bytes per layer (`32 * 3,345,412`), while actual allocation must be read from the backend:

- two layers: about 204.19 MiB before backend alignment effects;
- five layers: about 510.47 MiB;
- thirty layers: about 3,062.81 MiB.

These are projections, not measured VRAM claims. Each ramp report must record exact component offsets, exact buffer allocation, sampled peak GPU memory, settled memory, remaining physical reserve, and separately identified KV/state headroom. Allocation or cleanup growth stops the ramp.

### Configuration and compatibility

Default behavior remains unchanged when `EXPERTFLOW_LIVE_CACHE` is absent. Multi-layer mode uses:

- `EXPERTFLOW_LIVE_CACHE=1`
- `EXPERTFLOW_LIVE_CACHE_MODE=passthrough|blocking`
- `EXPERTFLOW_LIVE_CACHE_LAYERS=0,24` as an explicit, ascending, unique comma-separated list
- `EXPERTFLOW_LIVE_CACHE_LOG=<absolute Windows path>` for blocking mode
- `EXPERTFLOW_LIVE_CACHE_FORCE_EVICT=1` only for bounded diagnostic tests

The legacy `EXPERTFLOW_LIVE_CACHE_LAYER=24` spelling remains accepted as the exact one-layer C5 compatibility path. Supplying both singular and plural settings is an error. Empty elements, whitespace, ranges, duplicates, non-decimal values, values outside `0..29`, and non-ascending lists fail explicitly.

The probe generates CPU tensor overrides for exactly the three component names of every configured layer. Passthrough mode initializes configuration only and rejects log/forced-eviction settings. Blocking mode requires an absolute log path. Feature-off must restore the original source and scheduling path.

## Data path and invariants

For each enabled layer invocation:

1. identify the layer from the exact bound tensor metadata;
2. obtain the already-materialized authoritative eight IDs through the C5 boundary;
3. plan hits, protected LRU replacements, and generations in that layer's state;
4. blocking-copy every missing expert's three packed components into that layer's arena regions;
5. synchronize the destination backend once for the completed layer transfer set;
6. validate bytes, bounds, component ownership, layer ownership, generations, and all eight mappings;
7. publish the physical IDs and commit only that layer's state;
8. execute the unchanged CUDA operations directly from that layer's 32-slice arena views.

Failures cannot publish partial mappings. A layer cannot consume another layer's arena tensor, slot state, event, source binding, or selected-ID preparation marker. No GPU allocation/free, file I/O, string formatting, or heap allocation occurs per token.

## Event schema

Write bounded records at teardown, not in the hot path. Schema `1.2.0` extends C5 with:

- `token_index`
- `layer_id`
- `layer_access_sequence`
- `selected`
- `physical_slots`
- `hits`
- `blocking_misses`
- `loads` with expert, slot, replacement, and generation before/after
- `bytes_transferred`
- `blocking_duration_us`
- `final_resident_mapping`

The report aggregates event and demand counts, hits, misses, hit rate, transferred bytes, host-wall blocking duration, and blocking per generated token both globally and by layer. `blocking_duration_us` remains host-wall copy-plus-synchronization time; it is not CUDA-event latency or copy/compute overlap.

## Validation gates

At every stage, run cache-off and cache-on with the fixed general, code, and translation benchmark prompts, one warmup plus three measured repetitions. Require exact prompt/generated tokens, ordered router IDs, layer IDs, event counts, and causal ordering; deterministic cache decisions; stable host/GPU memory; complete process cleanup; and feature-off restoration.

Report prompt TPS, decode TPS, end-to-end time, TTFT, available p50/p95 token latency, exact counts, per-layer cache accounting, aggregate blocking time, actual arena size, peak VRAM, remaining reserve, and KV/state headroom. Treat system-wide GPU sampling separately from exact backend allocation.

Advance only when the current ramp point:

- preserves exactness and deterministic per-layer mappings;
- has stable allocation and cleanup;
- has no severe unexplained decode-TPS regression;
- reduces aggregate blocking sufficiently to leave a plausible path to the strongest stock no-OOM baseline;
- leaves measured reserve for ordinary runtime, KV, state, and desktop variance.

Stop before the next ramp on parity drift, stale/cross-layer mapping, incomplete bindings, repacking, per-token allocation, synchronization hazard, persistent memory growth, OOM, broad graph/allocator/kernel redesign, severe unexplained regression, or loss of disabled-default isolation. Passing all 30 layers does not authorize 64 slots; that requires a separate evidence-backed decision.

