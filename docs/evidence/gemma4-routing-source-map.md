# Gemma 4 routing telemetry source map

- **Inspection revision:** `ggml-org/llama.cpp@a7312ae94f801fc9c6786dc56e38df57b964f697` (release `b10002`)
- **Inspected:** 2026-07-14 PKT
- **Source verdict:** PASS
- **Overall 24-hour gate:** CONDITIONAL; parity-safe Vulkan telemetry passed, live cache still needs byte and transfer measurements

## Routing path

The initial live-HEAD inspection used `bf2c86ddc0685f580595954056c2e77ebabfab4f`. After the official `b10002` executable reported build commit `a7312ae94`, the full commit was resolved through GitHub's commit API and every cited routing boundary below was rechecked against the release-matched source. The cited line numbers and tensor contracts are identical.

1. `src/models/gemma4.cpp:100` loads the optional router tensor `ffn_gate_inp` with shape `[n_embd, n_expert]`; its presence marks an MoE layer.
2. `src/models/gemma4.cpp:316` computes Gemma 4 router logits from `attn_out`. At line 320, `build_lora_mm` produces `[n_expert, n_tokens]`, named `ffn_moe_logits-{layer}` by the graph callback.
3. `src/models/gemma4.cpp:323` passes those logits into `llm_graph_context::build_moe_ffn` together with `n_expert` and `n_expert_used`.
4. `src/llama-graph.cpp:1913` materializes `selected_experts`. When no indices are supplied by a caller, line 1915 calls `ggml_argsort_top_k(selection_probs, n_expert_used)`, producing `[n_expert_used, n_tokens]`.
5. `src/llama-graph.cpp:1918` invokes the existing graph callback with the stable name `ffn_moe_topk-{layer}`. Line 1929 uses the same indices to gather router weights with shape `[1, n_expert_used, n_tokens]`, named `ffn_moe_weights-{layer}`.
6. The selected indices remain authoritative inputs to the expert matrix operations at lines 1977, 1996, 2009, and 2098. Telemetry does not need to alter any of these edges.

## Evaluation-callback telemetry boundary

The public `llama_context_params.cb_eval` scheduler callback exposes the selected-expert tensors to a telemetry-only probe:

- During the callback's `ask=true` phase, return `true` only for tensor names beginning with `ffn_moe_topk-` and, optionally, `ffn_moe_weights-`.
- During the follow-up `ask=false` call, copy that small tensor to host with `ggml_backend_tensor_get`, capture a monotonic timestamp, parse the layer suffix, and emit one record per token column.
- The callback is already exercised by `examples/eval-callback/eval-callback.cpp` and `common/debug.cpp`; it does not require an allocator, model-format, router, or graph-semantics change.

Prompt evaluation may expose multiple token columns in one callback. The probe must associate those columns with the decode batch's absolute token range; generation normally contributes one column per decode. Tensor type and dimensions are asserted at runtime rather than assumed.

This boundary is not execution-neutral on every backend. In the pinned scheduler, no callback computes a whole backend split asynchronously. Installing a callback changes that path to synchronized graph views ending at each requested tensor. Stratified measurements show that the CUDA callback path changes greedy generated tokens for three of five prompts, even though each mode is independently repeatable. The same probe and workload preserve exact parity on the official b10002 Vulkan backend.

## Gate decision

Source checks 2 and 3 pass: the exact routing operation is identified, and the trace hook is bounded to an evaluation callback plus a small writer. The implementation is a separate probe executable and does not modify `build_moe_ffn`.

The real Q4 GGUF now loads and generates text. Tracing-disabled and tracing-enabled probe runs match all 38 prompt token IDs and 8 generated token IDs. The strict schema validates 1,350 router events over 45 decoded tokens and 30 MoE layers. An 8-slot static hotset estimates a 36.37% hit rate versus 31.87% for online LRU on this short trace.

The original one-prompt result did not establish backend-wide transparency. The later five-prompt CUDA matrix passed parity for only two prompts; a CPU control passed, isolating the failure to the CUDA observation path. The same five-prompt matrix passed exact parity on the pinned Vulkan backend.

For 3,840 fixed-prompt token/layer events, CUDA and Vulkan selected the same ordered top-8 set in 84.74% of events and overlapped on 99.26% of individual expert demands. Their aggregate prefill static-8 estimates differ by 0.0065 percentage points and LRU-8 by 0.0358 points. ExpertFlow therefore uses CUDA for inference/memory baselines and Vulkan for transparent GPU router telemetry. The overall live-cache gate remains conditional because expert byte size, transfer time, and end-to-end benefit are not measured.
