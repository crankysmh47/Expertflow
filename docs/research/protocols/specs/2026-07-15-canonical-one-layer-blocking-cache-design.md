# Canonical One-Layer Blocking Cache Design

**Status:** Approved for test-first implementation with eight coordinated replaceable slots on 2026-07-16.

## Objective

Prove that the canonical Observer v1 runtime can execute layer 24 with its true-router-selected packed Q4 experts loaded from the existing host representation into a reusable CUDA slot layout, including repeated replacement, while preserving exact prompt tokens, generated tokens, and logical router selections. This is a correctness and physical-feasibility proof, not a speed claim.

## Preserved baseline

- ExpertFlow base commits: `42a6b21` and `56d2ab4`.
- Canonical binary: `7ea12e0c44258bb2d75f99b7e180e7ef7cb0c7d3e285ca231fe26ad8c8c4932c`.
- Model: `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`.
- llama.cpp pin: `a7312ae94f801fc9c6786dc56e38df57b964f697`.
- Protected Observatory: `d846bdfcb1980dfc44d9f951e2824f58429f16d7`.
- Default remains `live_cache_enabled=false`. Cache-off runs use the accepted observer-enabled runtime and frozen decoding configuration.

## Selected layer and tensors

Layer 24 is a representative middle MoE layer and lies in the nine repeating layers offloaded by the canonical `-ngl 10` configuration. Its cache object has three coordinated components:

| Tensor | GGUF type and shape | Expert stride | Bytes per expert |
|---|---|---:|---:|
| `blk.24.ffn_gate_up_exps.weight` | Q4_0 `[2816, 1408, 128]` | 2,230,272 | 2,230,272 |
| `blk.24.ffn_down_exps.weight` | Q4_0 `[704, 2816, 128]` | 1,115,136 | 1,115,136 |
| `blk.24.ffn_down_exps.scale` | F32 `[128]` | 4 | 4 |

The encoded object is 3,345,412 bytes. The conservative aligned contract is 3,346,048 bytes. Q4_0 rows use 18-byte blocks for 32 elements. Gate/up strides are `nb=[18,1584,2230272]`; down strides are `nb=[18,396,1115136]`; scale stride is 4 bytes.

`llama-graph.cpp` binds both weight tensors and the same logical `ffn_moe_topk-24` tensor while constructing the two `MUL_MAT_ID` operations. The gate/up result is split into gate and up views, and the down operation consumes the second packed tensor. The scale vector is indexed by the same IDs. Therefore all three components must share one logical-to-physical mapping.

## Placement and integration boundary

Cache-off retains the canonical GPU placement. Cache-enabled initialization uses llama.cpp's existing tensor-buffer override contract to keep only the three layer-24 expert tensors on the CPU while the layer and its operations remain assigned to CUDA. This activates the existing host-weight `MUL_MAT_ID` scheduler path in `ggml/src/ggml-backend.cpp`, which already reads and synchronizes the authoritative selected IDs.

The cache implementation lives at that scheduler boundary. It preallocates destination tensor layouts once, loads direct Q4/F32 expert slices with blocking backend copies, and substitutes physical IDs only for the two layer-24 matrix operations and scale lookup. The canonical observer continues to receive and report logical IDs.

## Minimum slot count

Gemma selects eight experts per token and the unchanged CUDA `MUL_MAT_ID` kernels index one rank-3 tensor. One or two total slots cannot execute a top-8 event without either a mixed-storage kernel indirection or splitting/rebuilding the graph, both prohibited. The smallest exact arena is therefore eight coordinated physical slots.

Seven stable slots plus one replaceable slot can prove only C2/C3 on a controlled sequence. The canonical training pilot contains a layer-24 sequence in `train-translation-01`, decode indices 84-87, with fixed experts `2,3,5,32,38,58,91` and the eighth expert rotating through `33,34,44`.

That layout cannot pass C4 on arbitrary prompts: an event with two experts outside the fixed set cannot be represented, and the existing CUDA operation has no mixed slot/original-tensor fallback. The approved minimum C4 architecture is eight **replaceable** coordinated slots. Before each layer-24 operation it synchronously loads every missing member of the authoritative top-8 set, then remaps all eight logical IDs to physical slots. The user explicitly classified this as the minimum exact Gemma top-8 architecture, not a scope expansion.

## Configuration and events

The disabled-by-default environment contract is:

- `EXPERTFLOW_LIVE_CACHE=1`
- `EXPERTFLOW_LIVE_CACHE_LAYER=24`
- `EXPERTFLOW_LIVE_CACHE_FIXED_IDS=<seven comma-separated IDs>`
- `EXPERTFLOW_LIVE_CACHE_LOG=<absolute JSONL path>`
- `EXPERTFLOW_LIVE_CACHE_FORCE_EVICT=1` only for controlled tests

Each event records decode step, layer, logical expert, physical slot, hit/miss, bytes, host transfer start/end, blocking duration, eviction reason, and the complete resident mapping. Configuration, arena allocation, and log setup happen outside the capture/copy hot path. Overflow or an invalid contract fails explicitly.

## Correctness ladder

- C0: code compiled, variables unset; exact canonical observer parity.
- C1: enabled passthrough; configuration and telemetry active, original placement and execution unchanged.
- C2: layer-24 CPU override plus eight-slot arena with one predetermined replaceable expert loaded before inference.
- C3: deterministic prompts force at least two logical experts through the replaceable slot across repeated runs.
- C4: authoritative selected IDs drive all required blocking loads and remapping across eight replaceable slots; exact cache-off/cache-on parity across general, code, and translation, then the full seven-task suite.

Every level requires deterministic logical routing, no persistent host/GPU allocation growth, clean process exit, bounds checks, and feature-off restoration. Full ExpertFlow tests and judge replay run at the passing milestone.

## Stop conditions

Stop if tensor overrides do not traverse the selected-expert transfer path; direct packed slices require repacking; eight-slot remapping changes weights or logical routing; scale lookup cannot be isolated; an active slot can be replaced before execution completes; graph construction, CUDA kernels, or the general allocator require redesign; exact parity fails; or allocation growth/corruption appears.

No ML training, prediction, asynchronous copy, MTP, RL, multi-layer cache, or performance claim is part of this design.
