# Canonical One-Layer Cache Mapping

## Verdict

**C4 PASSED.** The user approved eight coordinated replaceable slots as the minimum exact Gemma top-8 architecture on 2026-07-16. The completed measurement and parity evidence is in [one-layer-c4-result.md](one-layer-c4-result.md). The default release remains cache-disabled.

## Baseline identity

- ExpertFlow: `56d2ab4cf96ce4b036fc9518b63158491551923a`
- canonical probe SHA-256: `7ea12e0c44258bb2d75f99b7e180e7ef7cb0c7d3e285ca231fe26ad8c8c4932c`
- model SHA-256: `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- llama.cpp: `a7312ae94f801fc9c6786dc56e38df57b964f697`
- protected Observatory: `d846bdfcb1980dfc44d9f951e2824f58429f16d7`
- focused baseline: 89 tests passed; canonical and protected worktrees clean

## Layer 24 object

| Component | Type | Shape | Strides | Per-expert bytes | Consumer |
|---|---|---|---|---:|---|
| `blk.24.ffn_gate_up_exps.weight` | Q4_0 | `[2816,1408,128]` | `[18,1584,2230272]` | 2,230,272 | fused gate/up `MUL_MAT_ID` |
| `blk.24.ffn_down_exps.weight` | Q4_0 | `[704,2816,128]` | `[18,396,1115136]` | 1,115,136 | down `MUL_MAT_ID` |
| `blk.24.ffn_down_exps.scale` | F32 | `[128]` | `[4]` | 4 | logical expert scale |

The packed object is 3,345,412 bytes and its aligned contract is 3,346,048 bytes. Cache-off places these tensors in the 4,534.75 MiB CUDA model buffer under `-ngl 10`. Cache-enabled placement can use the existing per-tensor CPU override; no loader rewrite is needed.

Both matrix operations receive the same `ffn_moe_topk-24` rank-2 I32 tensor. CUDA indexes expert slices through `src0->ne[2]` and the supplied IDs. Graph construction binds the source tensor addresses before execution. The existing scheduler can create a backend copy and already materializes authoritative IDs for host-resident MoE weights.

## Why one slot cannot reach C4

The selected-ID tensor contains eight unique logical experts per token. A reduced destination with one or two expert slices cannot accept the remaining logical IDs. The current kernel accepts one contiguous rank-3 source tensor; it cannot select some experts from the original tensor and one from an external slot. Supporting that would require a CUDA indirection change or splitting/rebuilding the graph, both explicit stop conditions.

A seven-fixed/one-replaceable proof exists in the training split: `train-translation-01`, layer 24, decode indices 84-87, fixed IDs `2,3,5,32,38,58,91`, rotating IDs `33,34,44`. It proves controlled replacement only. It cannot establish the requested true-router C4 suite.

## Constrained alternative

Use eight coordinated replaceable slots for the same single layer. At each event, block until all missing top-8 experts' three packed components are loaded, remap the eight logical IDs to slot IDs for the two matrix operations and scale, then execute. No prediction, async copy, multi-layer state, repacking, or general allocator redesign is needed.

Implementation completed test-first on the isolated branches. `live_cache_enabled=false` remains the default release state after C4; expansion requires a separate decision.
