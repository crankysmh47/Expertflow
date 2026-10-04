// Throwaway feasibility fixture: actual pristine DLLs, not product/native TPS evidence.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <windows.h>
#include <tlhelp32.h>

static std::atomic<unsigned long long> pdl{0}, classic{0}, errors{0};
static void CUPTIAPI callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void * data) {
    if (domain == CUPTI_CB_DOMAIN_STATE) {
        const auto * state = static_cast<const CUpti_StateData *>(data);
        fprintf(stderr, "CUPTI state %u result %d: %s\n", id, state->notification.result,
                state->notification.message ? state->notification.message : "no message");
        return;
    }
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
    const auto * info = static_cast<const CUpti_CallbackData *>(data);
    const bool ex = id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 ||
                    id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060;
    const bool ordinary = id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
                          id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000;
    if (!ex && !ordinary) return;
    if (info->callbackSite == CUPTI_API_EXIT) {
        if (info->functionReturnValue && *static_cast<const cudaError_t *>(info->functionReturnValue) != cudaSuccess) ++errors;
        return;
    }
    bool active = false;
    if (ex) {
        const auto * params = static_cast<const cudaLaunchKernelExC_v11060_params *>(info->functionParams);
        for (unsigned int i = 0; i < params->config->numAttrs; ++i) {
            const auto & attr = params->config->attrs[i];
            if (attr.id == cudaLaunchAttributeProgrammaticStreamSerialization && attr.val.programmaticStreamSerializationAllowed) active = true;
        }
    }
    if (active) ++pdl; else ++classic;
}

int main(int argc, char ** argv) {
    assert(argc == 2);
    CUpti_SubscriberHandle subscriber;
    CUptiResult subscribe = cuptiSubscribe(&subscriber, callback, nullptr);
    if (subscribe != CUPTI_SUCCESS) { fprintf(stderr, "CUPTI subscribe failed: %d\n", subscribe); return 3; }
    assert(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API) == CUPTI_SUCCESS);
    assert(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_STATE) == CUPTI_SUCCESS);
    FILE * file = fopen(argv[1], "wb"); assert(file);
    auto backend = ggml_backend_cuda_init(0); assert(backend);
    int shapes = 0, computes = 0;
    for (int width : {256, 768}) for (int rows : {17, 128, 129}) for (int tokens : {1, 3, 39}) {
        auto ctx = ggml_init({16*1024*1024, nullptr, true}); assert(ctx);
        auto weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, width, rows);
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
        auto norm = ggml_rms_norm(ctx, input, 1e-6f);
        auto mm = ggml_mul_mat(ctx, weights, norm);
        auto output = ggml_soft_max(ctx, mm);
        auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); assert(buffer);
        std::vector<float> wf(width*rows), x(width*tokens), result(rows*tokens);
        for (size_t i = 0; i < wf.size(); ++i) wf[i] = (int((i*37+i/width*19)%509)-254)/127.0f;
        std::vector<unsigned char> quant(ggml_nbytes(weights));
        assert(ggml_quantize_chunk(GGML_TYPE_Q6_K, wf.data(), quant.data(), 0, rows, width, nullptr) == quant.size());
        ggml_backend_tensor_set(weights, quant.data(), 0, quant.size());
        for (int iteration = 0; iteration < 3; ++iteration) {
            for (size_t i = 0; i < x.size(); ++i) x[i] = (int((i*71+i/width*13+iteration*17)%251)-125)/61.0f;
            ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
            assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(float));
            for (float value : result) assert(std::isfinite(value) && value >= 0 && value <= 1);
            assert(fwrite(result.data(), sizeof(float), result.size(), file) == result.size());
            ++computes;
        }
        ++shapes;
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    HANDLE modules = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
    assert(modules != INVALID_HANDLE_VALUE);
    MODULEENTRY32 entry; entry.dwSize = sizeof(entry);
    assert(Module32First(modules, &entry));
    do { fprintf(stderr, "MODULE\t%s\t%s\n", entry.szModule, entry.szExePath); } while (Module32Next(modules, &entry));
    CloseHandle(modules);
    ggml_backend_free(backend);
    assert(fclose(file) == 0);
    CUptiResult detached = cuptiUnsubscribe(subscriber);
    const char * detach_message = nullptr;
    cuptiGetResultString(detached, &detach_message);
    fprintf(stderr, "CUPTI unsubscribe: %d (%s)\n", detached, detach_message ? detach_message : "unknown");
    printf("{\"shapes\":%d,\"computes\":%d,\"pdl_launches\":%llu,\"classic_launches\":%llu,\"cuda_launch_errors\":%llu}\n",
        shapes, computes, pdl.load(), classic.load(), errors.load());
    return detached != CUPTI_SUCCESS ? 4 : (errors ? 2 : 0);
}
