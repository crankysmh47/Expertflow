#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cstdio>
#include <vector>

// Emits actual expert graph outputs for stock/candidate bitwise comparison.
int main(int argc, char ** argv) {
    assert(argc == 2);
    FILE * file = fopen(argv[1], "wb");
    assert(file);
    auto cpu = ggml_backend_cpu_init();
    for (int threads : {1, 12}) {
        ggml_backend_cpu_set_n_threads(cpu, threads);
        for (int rows : {1, 15, 16, 17, 33, 257}) {
            for (int tokens : {1, 3}) {
                for (int width : {256, 768}) {
                    auto ctx = ggml_init({16*1024*1024, nullptr, true});
                    auto weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, width, rows, 3);
                    auto input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 2, tokens);
                    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, tokens);
                    auto output = ggml_mul_mat_id(ctx, weights, input, ids);
                    auto graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, output);
                    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
                    assert(buffer);
                    std::vector<float> wf(width*rows*3), x(width*2*tokens);
                    for (size_t i = 0; i < wf.size(); ++i) {
                        wf[i] = (int((i*37 + i/width*19) % 509) - 254) / 127.0f;
                    }
                    for (size_t i = 0; i < x.size(); ++i) {
                        x[i] = (int((i*71 + i/width*13) % 251) - 125) / 61.0f;
                    }
                    std::vector<unsigned char> quant(ggml_nbytes(weights));
                    assert(ggml_quantize_chunk(GGML_TYPE_Q6_K, wf.data(), quant.data(),
                                              0, rows*3, width, nullptr) == quant.size());
                    std::vector<int32_t> selected(2*tokens);
                    for (int t = 0; t < tokens; ++t) {
                        selected[2*t] = t % 3;
                        selected[2*t+1] = (t+2) % 3;
                    }
                    ggml_backend_tensor_set(weights, quant.data(), 0, quant.size());
                    ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
                    ggml_backend_tensor_set(ids, selected.data(), 0, selected.size()*sizeof(int32_t));
                    assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
                    std::vector<float> result(rows*2*tokens);
                    ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(float));
                    assert(fwrite(result.data(), sizeof(float), result.size(), file) == result.size());
                    ggml_backend_buffer_free(buffer);
                    ggml_free(ctx);
                }
            }
        }
    }
    ggml_backend_free(cpu);
    assert(fclose(file) == 0);
    puts("48 Q6 expert graph fixtures completed");
}
