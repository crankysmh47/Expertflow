# Exact upstream operation excerpts

All line numbers refer to a7312ae94f801fc9c6786dc56e38df57b964f697, not the current fork worktree. Git blob IDs/raw SHA-256 digests and complete sources are preserved in source-inspection.json and supplemental-source-objects.json plus their archives.

Display lines omit trailing whitespace; archived source bytes are unchanged.

## CPU and CUDA quantized operations

`ggml/src/ggml-cpu/ggml-cpu.c:239-247`

```cpp
239:     [GGML_TYPE_Q4_0] = {
240:         .from_float               = quantize_row_q4_0,
241:         .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
242:         .vec_dot_type             = GGML_TYPE_Q8_0,
243: #if defined (__ARM_FEATURE_MATMUL_INT8)
244:         .nrows                    = 2,
245: #else
246:         .nrows                    = 1,
247: #endif
```

## CPU Q6 activation type

`ggml/src/ggml-cpu/ggml-cpu.c:326-334`

```cpp
326:     [GGML_TYPE_Q6_K] = {
327:         .from_float               = quantize_row_q6_K,
328:         .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
329:         .vec_dot_type             = GGML_TYPE_Q8_K,
330: #if defined (__ARM_FEATURE_MATMUL_INT8)
331:         .nrows                    = 2,
332: #else
333:         .nrows                    = 1,
334: #endif
```

## Layer placement selector

`src/llama-model.cpp:1293-1304`

```cpp
1293:     const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
1294:     const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
1295:     auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
1296:         const bool is_swa = il < n_layer_all && hparams.is_swa(il);
1297:         if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
1298:             LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
1299:             return {cpu_dev, &pimpl->cpu_buft_list};
1300:         }
1301:         const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
1302:         auto * dev = devices.at(layer_gpu).dev;
1303:         LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
1304:         return {dev, &pimpl->gpu_buft_list.at(dev)};
```

## Expert CPU override

`common/arg.cpp:2472-2495`

```cpp
2472:     add_opt(common_arg(
2473:         {"-cmoe", "--cpu-moe"},
2474:         "keep all Mixture of Experts (MoE) weights in the CPU",
2475:         [](common_params & params) {
2476:             params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
2477:         }
2478:     ).set_env("LLAMA_ARG_CPU_MOE"));
2479:     add_opt(common_arg(
2480:         {"-ncmoe", "--n-cpu-moe"}, "N",
2481:         "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU",
2482:         [](common_params & params, int value) {
2483:             if (value < 0) {
2484:                 throw std::invalid_argument("invalid value");
2485:             }
2486:             for (int i = 0; i < value; ++i) {
2487:                 // keep strings alive and avoid leaking memory by storing them in a static vector
2488:                 static std::list<std::string> buft_overrides;
2489:                 buft_overrides.push_back(llm_ffn_exps_block_regex(i));
2490:                 params.tensor_buft_overrides.push_back({buft_overrides.back().c_str(), ggml_backend_cpu_buffer_type()});
2491:             }
2492:         }
2493:     ).set_env("LLAMA_ARG_N_CPU_MOE"));
2494:     GGML_ASSERT(params.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
2495:     add_opt(common_arg(
```

## First matching buffer override

`src/llama-model-loader.cpp:1162-1188`

```cpp
1162:         ggml_backend_buffer_type_t buft = nullptr;
1163:
1164:         // check overrides
1165:         if (tensor_buft_overrides) {
1166:             std::string tensor_name = tn.str();
1167:             for (const auto * overrides = tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
1168:                 std::regex pattern(overrides->pattern);
1169:                 if (std::regex_search(tensor_name, pattern)) {
1170:                     if (overrides->buft == ggml_backend_cpu_buffer_type()) {
1171:                         // when overriding to a CPU buffer, consider the extra buffer types
1172:                         buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);
1173:                         if (use_mmap) {
1174:                             static std::once_flag once;
1175:                             std::call_once(once, [] {
1176:                                 LLAMA_LOG_WARN("llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --no-mmap for better performance\n");
1177:                             });
1178:                         }
1179:                     } else {
1180:                         buft = overrides->buft;
1181:                     }
1182:
1183:                     LLAMA_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n",
1184:                             tensor_name.c_str(),
1185:                             ggml_nbytes(t_meta) / 1024 / 1024, ggml_type_name(t_meta->type),
1186:                             ggml_backend_buft_name(buft));
1187:                     break;
1188:                 }
```

## CUDA routed dispatch depends on shape

`ggml/src/ggml-cuda/ggml-cuda.cu:1785-1810`

```cpp
1785:     // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
1786:     if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
1787:         static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
1788:         if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
1789:             if (ggml_is_quantized(src0->type)) {
1790:                 const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
1791:                 if (ne2 <= mmvq_mmid_max) {
1792:                     ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
1793:                     return;
1794:                 }
1795:             } else {
1796:                 if (GGML_CUDA_CC_IS_AMD(cc)) {
1797:                     ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
1798:                     return;
1799:                 }
1800:             }
1801:         }
1802:
1803:         if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
1804:             ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
1805:             return;
1806:         }
1807:
1808:         if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
1809:             ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
1810:             return;
```

## CUDA Q8_1 activation conversion

`ggml/src/ggml-cuda/mmvq.cu:1222-1233`

```cpp
1222:     const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
1223:     ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
1224:     {
1225:         const int64_t s11 = src1->nb[1] / ts_src1;
1226:         const int64_t s12 = src1->nb[2] / ts_src1;
1227:         const int64_t s13 = src1->nb[3] / ts_src1;
1228:         quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
1229:     }
1230:
1231:     const int64_t s01 = src0->nb[1] / ts_src0;
1232:     const int64_t s11 = ne10_padded / QK8_1;
1233:     const int64_t s1  =  dst->nb[1] / ts_dst;
```

## Attention fused branch

`src/llama-graph.cpp:2405-2429`

```cpp
2405:     ggml_tensor * cur;
2406:
2407:     const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
2408:     if (use_flash_attn) {
2409:         GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");
2410:
2411:         if (v_trans) {
2412:             v = ggml_transpose(ctx0, v);
2413:         }
2414:
2415:         // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
2416:         if (k->type == GGML_TYPE_F32) {
2417:             k = ggml_cast(ctx0, k, GGML_TYPE_F16);
2418:         }
2419:
2420:         if (v->type == GGML_TYPE_F32) {
2421:             v = ggml_cast(ctx0, v, GGML_TYPE_F16);
2422:         }
2423:
2424:         cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
2425:                                   hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
2426:         res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
2427:
2428:         ggml_flash_attn_ext_add_sinks(cur, sinks);
2429:         ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);
```

## Attention separate branch

`src/llama-graph.cpp:2449-2455`

```cpp
2449:     } else {
2450:         ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
2451:         cb(kq, "kq", il);
2452:
2453:         // note: this op tends to require high floating point range
2454:         //       while for some models F16 is enough, for others it is not, so we default to F32 here
2455:         ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
```

## Attention separate softmax

`src/llama-graph.cpp:2483-2506`

```cpp
2483:
2484:         kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
2485:         ggml_soft_max_add_sinks(kq, sinks);
2486:         cb(kq, "kq_soft_max", il);
2487:
2488:         if (!v_trans) {
2489:             // note: avoid this branch
2490:             v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
2491:             cb(v, "v_cont", il);
2492:         }
2493:
2494:         ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
2495:         cb(kqv, "kqv", il);
2496:
2497:         // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
2498:         if (v_mla) {
2499:             kqv = ggml_mul_mat(ctx0, v_mla, kqv);
2500:             cb(kqv, "kqv_mla", il);
2501:         }
2502:
2503:         cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
2504:
2505:         // recombine streams
2506:         cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
```

## Attention mask type

`src/llama-graph.cpp:31-40`

```cpp
31:     const auto n_kv     = mctx->get_n_kv();
32:     const auto n_tokens = ubatch.n_tokens;
33:     const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
34:
35:     // flash attention requires an f16 mask
36:     const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
37:
38:     ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
39:     ggml_set_input(res);
40:     ggml_set_name(res, "attn_inp_kq_mask");
```

## Fused attention shape dispatch

`ggml/src/ggml-cuda/fattn.cu:9-34`

```cpp
9: template <int DKQ, int DV, int ncols2>
10: static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
11:     const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
12:     const ggml_tensor * Q = dst->src[0];
13:
14:     if constexpr (ncols2 <= 8) {
15:         if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
16:             ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
17:             return;
18:         }
19:     }
20:
21:     if constexpr (ncols2 <= 16) {
22:         if (Q->ne[1] <= 16/ncols2) {
23:             ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
24:             return;
25:         }
26:     }
27:
28:     if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
29:             (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
30:         ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
31:         return;
32:     }
33:
34:     ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
```

## Effective batch caps and output reservation

`src/llama-context.cpp:238-246`

```cpp
238:     // with causal attention, the batch size is limited by the context size
239:     cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;
240:
241:     cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);
242:
243:     cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
244:
245:     cparams.op_offload = params.op_offload;
246:     cparams.kv_unified = params.kv_unified;
```

## Batch passed to memory initialization

`src/llama-context.cpp:1750-1760`

```cpp
1750:     llama_memory_context_ptr mctx;
1751:
1752:     while (true) {
1753:         mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
1754:         if (!mctx) {
1755:             return -2;
1756:         }
1757:
1758:         switch (mctx->get_status()) {
1759:             case LLAMA_MEMORY_STATUS_SUCCESS:
1760:                 {
```

## Microbatch splitting

`src/llama-batch.cpp:476-508`

```cpp
476: llama_ubatch llama_batch_allocr::split_simple(uint32_t n_ubatch) {
477:     // find the first unused token
478:     uint32_t cur_idx = 0;
479:     while (cur_idx < used.size() && used[cur_idx]) {
480:         ++cur_idx;
481:     }
482:
483:     // we are done
484:     if (cur_idx >= used.size()) {
485:         return {};
486:     }
487:
488:     std::vector<int32_t> idxs;
489:
490:     while (true) {
491:         idxs.push_back(cur_idx);
492:
493:         used[cur_idx] = true;
494:         ++n_used;
495:
496:         ++cur_idx;
497:
498:         if (cur_idx >= used.size()) {
499:             break;
500:         }
501:
502:         if (idxs.size() >= n_ubatch) {
503:             break;
504:         }
505:     }
506:
507:     return ubatch_add(idxs, idxs.size(), false);
508: }
```

## Serial non-speculative decode

`tools/server/server-context.cpp:448-457`

```cpp
448:     void handle_last_sampled_token(server_batch & batch) {
449:         bool add_ok = true;
450:         if (spec_draft.empty()) {
451:             // no speculative decoding
452:             i_batch = batch.size();
453:
454:             add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true);
455:
456:             SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
457:                     sampled, n_ctx, prompt.n_tokens(), truncated);
```

## Server batch caps

`tools/server/server-context.cpp:3011-3025`

```cpp
3011:         // update the batch with the sampled/drafted tokens
3012:         iterate(generating, [&](server_slot & slot) {
3013:             slot.handle_last_sampled_token(batch);
3014:         });
3015:
3016:         // process in chunks of params.n_batch
3017:         int32_t n_batch  = llama_n_batch(ctx_tgt);
3018:         int32_t n_ubatch = llama_n_ubatch(ctx_tgt);
3019:
3020:         auto & alora_scale       = batch.alora_scale;
3021:         auto & alora_disabled_id = batch.alora_disabled_id;
3022:
3023:         // next, batch any pending prompts without exceeding n_batch
3024:         if (params_base.cont_batching || batch.size() == 0) {
3025:             bool add_ok = true; // false means the batch is full, skip remaining slots
```
