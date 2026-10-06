#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cassert>

int main() {
    auto cpu = ggml_backend_cpu_init();
    auto sched = ggml_backend_sched_new(&cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_init_params params = { 1024*1024, nullptr, true };
    auto ctx = ggml_init(params);
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    ggml_set_name(input, "input"); ggml_set_input(input);
    auto output = ggml_add(ctx, input, input);
    ggml_set_name(output, "ffn_moe_gate_up-0"); ggml_set_output(output);
    auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
    assert(ggml_backend_sched_alloc_graph(sched, graph));
    float values[64]; for (auto & value : values) { value = 1; }
    ggml_backend_tensor_set(input, values, 0, sizeof(values));
    for (int phase = 1; phase <= 3; ++phase) {
        ggml_backend_sched_profile_set_phase(sched, phase);
        ggml_backend_sched_profile_note_tokens(sched, phase == 2 ? 39 : 1);
        assert(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_tensor_get(output, values, 0, sizeof(values));
    for (auto value : values) { assert(value == 2); }
    ggml_backend_sched_free(sched); ggml_free(ctx); ggml_backend_free(cpu);
}
