# Gemma 4 routing telemetry source map

- **Inspection revision:** `ggml-org/llama.cpp@bf2c86ddc0685f580595954056c2e77ebabfab4f`
- **Inspected:** 2026-07-14 PKT
- **Source verdict:** PASS
- **Overall 24-hour gate:** PENDING real Q4 load, trace parity, and locality evidence

## Routing path

1. `src/models/gemma4.cpp:100` loads the optional router tensor `ffn_gate_inp` with shape `[n_embd, n_expert]`; its presence marks an MoE layer.
2. `src/models/gemma4.cpp:316` computes Gemma 4 router logits from `attn_out`. At line 320, `build_lora_mm` produces `[n_expert, n_tokens]`, named `ffn_moe_logits-{layer}` by the graph callback.
3. `src/models/gemma4.cpp:323` passes those logits into `llm_graph_context::build_moe_ffn` together with `n_expert` and `n_expert_used`.
4. `src/llama-graph.cpp:1913` materializes `selected_experts`. When no indices are supplied by a caller, line 1915 calls `ggml_argsort_top_k(selection_probs, n_expert_used)`, producing `[n_expert_used, n_tokens]`.
5. `src/llama-graph.cpp:1918` invokes the existing graph callback with the stable name `ffn_moe_topk-{layer}`. Line 1929 uses the same indices to gather router weights with shape `[1, n_expert_used, n_tokens]`, named `ffn_moe_weights-{layer}`.
6. The selected indices remain authoritative inputs to the expert matrix operations at lines 1977, 1996, 2009, and 2098. Telemetry does not need to alter any of these edges.

## Minimal telemetry boundary

The existing public `llama_context_params.cb_eval` scheduler callback is sufficient for a telemetry-only probe:

- During the callback's `ask=true` phase, return `true` only for tensor names beginning with `ffn_moe_topk-` and, optionally, `ffn_moe_weights-`.
- During the follow-up `ask=false` call, copy that small tensor to host with `ggml_backend_tensor_get`, capture a monotonic timestamp, parse the layer suffix, and emit one record per token column.
- The callback is already exercised by `examples/eval-callback/eval-callback.cpp` and `common/debug.cpp`; it does not require an allocator, scheduler, model-format, router, or graph-semantics change.

Prompt evaluation may expose multiple token columns in one callback. The probe must associate those columns with the decode batch's absolute token range; generation normally contributes one column per decode. Tensor type and dimensions will be asserted at runtime rather than assumed.

## Gate decision

Source checks 2 and 3 pass: the exact routing operation is identified, and the trace hook is bounded to an evaluation callback plus a small writer. The first implementation should be a separate probe executable or narrowly scoped callback integration, not a modification to `build_moe_ffn`.

The overall gate remains pending until:

- the pinned Q4 GGUF loads and generates text;
- uninstrumented and callback-instrumented deterministic token sequences match;
- trace records validate against the agreed schema; and
- measured locality supports at least one useful cache-policy result.
