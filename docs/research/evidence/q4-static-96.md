# What static-96 means

`static-96` is a fixed per-layer expert placement used by the Observatory simulator. It is not a live cache and it is not 96 global slots.

## Capacity and scope

- `96` means 96 resident expert slots in each target MoE layer.
- The current target is layers 0–20: 21 layers left on CPU by the measured 10-layer offload profile.
- Total projected slots: `96 × 21 = 2,016`.
- Each layer has 128 experts, so static-96 leaves 32 experts per target layer outside the fixed resident set.
- The placement is layer-partitioned. An expert ID in layer 3 and the same numeric expert ID in layer 4 occupy different objects and different slots.

The exact encoded object is 3,345,412 bytes. The conservative source-derived CUDA slot is 3,346,048 bytes after Q4 row-end padding and 128-byte tensor alignment. Therefore:

```text
3,346,048 bytes/slot × 96 slots/layer × 21 layers
= 6,745,632,768 bytes
= 6,433.136719 MiB
```

The measured configurable envelope is 7,234 MiB after a separate 1,024 MiB safety reserve. Static-96 leaves approximately 800.863 MiB of that configurable envelope. This establishes arithmetic fit for the projected slot arena, not a measured live allocation. Slot metadata, staging buffers, allocator fragmentation, and copy/compute workspace are not yet allocated or measured.

## How the hot set is selected

For each target layer independently:

1. Count selected-expert demands in the frozen training traces.
2. Sort by descending frequency.
3. Break equal-frequency ties by ascending expert ID.
4. Freeze the first 96 expert IDs before validation/test evaluation.

The current expanded result fits on 31 parity-safe training conversations. `train-translation-02` is excluded explicitly because two trace-off/trace-on pairs produced the same deterministic generated-token mismatch. Validation and test events never affect the fixed residents.

## Which metric is current

The current conservative number is decode evaluation, not a combined prefill/decode average:

- Fit phase: training prefill
- Evaluation phase: held-out decode
- Evaluation set: 8 untouched validation/test conversations across 8 domains
- Target: layers 0–20
- Static-96 hit rate: 87.57%
- Reset-per-conversation LRU hit rate: 86.34%

Prefill is reported separately at 95.85% static versus 87.48% LRU. It is useful evidence about initial working-set formation but is not substituted for decode behavior.

## Why this is not the earlier 36.37%

The 36.37% result came from one short Q4 trace, an eight-slot static hotset, all 30 MoE layers, and in-sample fitting/evaluation over the same prompt and continuation. It was an early feasibility sample.

Static-96 differs on every important axis:

| Property | Earlier 36.37% | Current static-96 |
| --- | --- | --- |
| Capacity | 8 slots/layer | 96 slots/layer |
| Workload | One prompt | 31 training and 8 held-out conversations |
| Fit/evaluation | Same trace | Conversation-level held out |
| Layer scope | All 30 MoE layers | Target layers 0–20 |
| Phase | Combined short trace | Train prefill, evaluate decode |
| Current hit rate | Superseded | 87.57% decode |

The larger number is mainly a larger physical budget, not evidence that the old static-8 claim improved. The expanded decode advantage over LRU is only 1.23 percentage points and is domain-dependent. `live_cache_enabled=false`.
