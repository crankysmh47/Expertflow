#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr const char * schema_version = "1.0.0";
constexpr const char * topk_prefix = "ffn_moe_topk-";
constexpr const char * llama_release = "b10002";
constexpr const char * llama_revision = "a7312ae94f801fc9c6786dc56e38df57b964f697";

enum class trace_mode {
    full,
    empty,
    disabled,
};

struct options {
    std::string model_path;
    std::string trace_path;
    std::string tokens_path;
    std::string prompt = "Explain in three concise sentences why a cache miss can stall sparse MoE inference.";
    int n_predict = 16;
    int n_gpu_layers = 0;
    int threads = 12;
    trace_mode mode = trace_mode::full;
    std::string capture_tensor;
    std::string capture_output;
    int capture_token_index = -1;
};

void usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s -m MODEL --tokens FILE [--trace FILE | --no-trace] "
        "[--trace-mode full|empty|disabled] "
        "[-n TOKENS] [-ngl LAYERS] [--threads N] "
        "[--capture-tensor NAME --capture-token-index INDEX --capture-output FILE] "
        "[PROMPT]\n",
        program);
}

bool parse_int(const char * value, int & output) {
    try {
        const std::string text(value);
        std::size_t consumed = 0;
        const long parsed = std::stol(text, &consumed);
        if (consumed != text.size() || parsed < 0 ||
            parsed > std::numeric_limits<int>::max()) {
            return false;
        }
        output = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_trace_mode(const char * value, trace_mode & output) {
    if (std::strcmp(value, "full") == 0) {
        output = trace_mode::full;
        return true;
    }
    if (std::strcmp(value, "empty") == 0) {
        output = trace_mode::empty;
        return true;
    }
    if (std::strcmp(value, "disabled") == 0) {
        output = trace_mode::disabled;
        return true;
    }
    return false;
}

bool parse_options(int argc, char ** argv, options & result) {
    int index = 1;
    for (; index < argc; ++index) {
        const char * argument = argv[index];
        if (std::strcmp(argument, "-m") == 0 && index + 1 < argc) {
            result.model_path = argv[++index];
        } else if (std::strcmp(argument, "--trace") == 0 && index + 1 < argc) {
            result.trace_path = argv[++index];
        } else if (std::strcmp(argument, "--tokens") == 0 && index + 1 < argc) {
            result.tokens_path = argv[++index];
        } else if (std::strcmp(argument, "--no-trace") == 0) {
            result.mode = trace_mode::disabled;
        } else if (std::strcmp(argument, "--trace-mode") == 0 && index + 1 < argc) {
            if (!parse_trace_mode(argv[++index], result.mode)) {
                return false;
            }
        } else if (std::strcmp(argument, "-n") == 0 && index + 1 < argc) {
            if (!parse_int(argv[++index], result.n_predict)) {
                return false;
            }
        } else if (std::strcmp(argument, "-ngl") == 0 && index + 1 < argc) {
            if (!parse_int(argv[++index], result.n_gpu_layers)) {
                return false;
            }
        } else if (std::strcmp(argument, "--threads") == 0 && index + 1 < argc) {
            if (!parse_int(argv[++index], result.threads) || result.threads == 0) {
                return false;
            }
        } else if (std::strcmp(argument, "--capture-tensor") == 0 && index + 1 < argc) {
            result.capture_tensor = argv[++index];
        } else if (std::strcmp(argument, "--capture-token-index") == 0 && index + 1 < argc) {
            if (!parse_int(argv[++index], result.capture_token_index)) {
                return false;
            }
        } else if (std::strcmp(argument, "--capture-output") == 0 && index + 1 < argc) {
            result.capture_output = argv[++index];
        } else if (argument[0] == '-') {
            return false;
        } else {
            break;
        }
    }

    if (index < argc) {
        result.prompt = argv[index++];
        for (; index < argc; ++index) {
            result.prompt += " ";
            result.prompt += argv[index];
        }
    }

    const bool capture_requested =
        !result.capture_tensor.empty() ||
        result.capture_token_index >= 0 ||
        !result.capture_output.empty();
    const bool capture_complete =
        !result.capture_tensor.empty() &&
        result.capture_token_index >= 0 &&
        !result.capture_output.empty();
    const bool full_trace = result.mode == trace_mode::full;

    return !result.model_path.empty() && !result.tokens_path.empty() &&
           (full_trace ? !result.trace_path.empty() : result.trace_path.empty()) &&
           (!capture_requested || (capture_complete && full_trace));
}

struct trace_state {
    std::ofstream output;
    std::ofstream capture_output;
    std::vector<llama_token> batch_tokens;
    std::string phase;
    std::string error;
    std::vector<std::string> observed_tensor_names;
    std::vector<std::string> observation_tensor_names;
    std::uint64_t base_token_index = 0;
    std::uint64_t forward_id = 0;
    std::uint64_t hook_order = 0;
    std::uint64_t callback_asks = 0;
    std::uint64_t selected_asks = 0;
    std::uint64_t callback_observations = 0;
    std::string capture_tensor;
    std::uint64_t capture_token_index = 0;
    bool capture_enabled = false;
    bool capture_written = false;
    bool active_forward = false;

    trace_state(
        const std::string & path,
        bool enabled,
        const std::string & next_capture_tensor,
        int next_capture_token_index,
        const std::string & next_capture_path) {
        if (enabled) {
            output.open(path, std::ios::out | std::ios::trunc);
        }
        if (!next_capture_tensor.empty()) {
            capture_tensor = next_capture_tensor;
            capture_token_index = static_cast<std::uint64_t>(next_capture_token_index);
            capture_enabled = true;
            capture_output.open(next_capture_path, std::ios::out | std::ios::trunc);
        }
    }

    void begin_forward(
        const llama_batch & batch,
        const char * next_phase,
        std::uint64_t next_base_token_index) {
        batch_tokens.assign(batch.token, batch.token + batch.n_tokens);
        phase = next_phase;
        base_token_index = next_base_token_index;
        active_forward = true;
    }

    void end_forward() { active_forward = false; }
};

bool starts_with(const char * value, const char * prefix) {
    return std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

int parse_layer_id(const char * name) {
    const char * suffix = name + std::strlen(topk_prefix);
    if (*suffix == '\0') {
        return -1;
    }
    char * end = nullptr;
    const long value = std::strtol(suffix, &end, 10);
    if (*end != '\0' || value < 0 || value > 65535) {
        return -1;
    }
    return static_cast<int>(value);
}

bool router_trace_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<trace_state *>(user_data);
    const char * name = ggml_get_name(tensor);
    const bool selected_experts = starts_with(name, topk_prefix);
    const bool capture_tensor =
        state.capture_enabled && state.capture_tensor == name;
    const bool capture_forward =
        state.active_forward &&
        state.capture_token_index >= state.base_token_index &&
        state.capture_token_index < state.base_token_index + state.batch_tokens.size();
    if (ask) {
        ++state.callback_asks;
        const std::string tensor_name(name);
        const bool routing_candidate =
            tensor_name.find("moe") != std::string::npos ||
            tensor_name.find("ffn") != std::string::npos ||
            tensor_name.find("top") != std::string::npos;
        if (routing_candidate && state.observed_tensor_names.size() < 100) {
            state.observed_tensor_names.push_back(tensor_name);
        }
        if (selected_experts && state.active_forward) {
            ++state.selected_asks;
        }
        return (selected_experts && state.active_forward) ||
               (capture_tensor && capture_forward);
    }
    ++state.callback_observations;
    if (state.observation_tensor_names.size() < 100) {
        state.observation_tensor_names.emplace_back(name);
    }
    if (capture_tensor && capture_forward) {
        const std::uint64_t local_token_index =
            state.capture_token_index - state.base_token_index;
        if (!ggml_is_contiguous(tensor) || tensor->ne[0] <= 0 ||
            tensor->ne[1] != static_cast<std::int64_t>(state.batch_tokens.size()) ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
            (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32)) {
            state.error = std::string("unexpected capture tensor contract for ") + name;
            return false;
        }

        const std::size_t value_count = static_cast<std::size_t>(tensor->ne[0]);
        const std::size_t element_size = sizeof(std::uint32_t);
        const char * capture_type = tensor->type == GGML_TYPE_F32 ? "f32" : "i32";
        const std::size_t byte_offset =
            static_cast<std::size_t>(local_token_index) * value_count * element_size;
        const std::size_t byte_count = value_count * element_size;
        std::vector<std::uint8_t> bytes(byte_count);
        ggml_backend_tensor_get(tensor, bytes.data(), byte_offset, byte_count);

        state.capture_output
            << "{\"schema_version\":\"" << schema_version
            << "\",\"tensor_name\":\"" << name
            << "\",\"tensor_type\":\"" << capture_type
            << "\",\"token_index\":" << state.capture_token_index
            << ",\"token_id\":" << state.batch_tokens[static_cast<std::size_t>(local_token_index)]
            << ",\"forward_id\":" << state.forward_id
            << ",\"dimensions\":[" << tensor->ne[0] << ',' << tensor->ne[1]
            << ',' << tensor->ne[2] << ',' << tensor->ne[3]
            << "],\"byte_count\":" << byte_count
            << ",\"values\":[";
        if (tensor->type == GGML_TYPE_F32) {
            const float * values = reinterpret_cast<const float *>(bytes.data());
            state.capture_output << std::setprecision(std::numeric_limits<float>::max_digits10);
            for (std::size_t value = 0; value < value_count; ++value) {
                if (value != 0) {
                    state.capture_output << ',';
                }
                state.capture_output << values[value];
            }
        } else {
            const std::int32_t * values = reinterpret_cast<const std::int32_t *>(bytes.data());
            for (std::size_t value = 0; value < value_count; ++value) {
                if (value != 0) {
                    state.capture_output << ',';
                }
                state.capture_output << values[value];
            }
        }
        state.capture_output << "]}\n";
        if (!state.capture_output) {
            state.error = "failed to write capture output";
            return false;
        }
        state.capture_written = true;
    }

    if (!selected_experts) {
        return true;
    }

    const int layer_id = parse_layer_id(name);
    if (layer_id < 0 || tensor->type != GGML_TYPE_I32 ||
        tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
        !ggml_is_contiguous(tensor)) {
        state.error = std::string("unexpected router tensor contract for ") + name +
            ": layer=" + std::to_string(layer_id) +
            ", type=" + std::to_string(static_cast<int>(tensor->type)) +
            ", dims=" + std::to_string(ggml_n_dims(tensor)) +
            ", ne0=" + std::to_string(tensor->ne[0]) +
            ", ne1=" + std::to_string(tensor->ne[1]) +
            ", contiguous=" + std::to_string(ggml_is_contiguous(tensor));
        return false;
    }
    if (static_cast<std::size_t>(tensor->ne[1]) != state.batch_tokens.size()) {
        state.error = std::string("router token columns do not match active batch for ") + name +
            ": columns=" + std::to_string(tensor->ne[1]) +
            ", batch_tokens=" + std::to_string(state.batch_tokens.size());
        return false;
    }

    std::vector<std::int32_t> expert_ids(
        static_cast<std::size_t>(tensor->ne[0] * tensor->ne[1]));
    ggml_backend_tensor_get(
        tensor, expert_ids.data(), 0, expert_ids.size() * sizeof(expert_ids[0]));

    for (std::int64_t token_column = 0; token_column < tensor->ne[1]; ++token_column) {
        state.output
            << "{\"schema_version\":\"" << schema_version
            << "\",\"request_id\":\"req-001\""
            << ",\"conversation_id\":\"conv-001\""
            << ",\"turn_index\":0"
            << ",\"phase\":\"" << state.phase << "\""
            << ",\"forward_id\":" << state.forward_id
            << ",\"hook_order\":" << state.hook_order++
            << ",\"token_index\":" << state.base_token_index + token_column
            << ",\"token_id\":" << state.batch_tokens[static_cast<std::size_t>(token_column)]
            << ",\"layer_id\":" << layer_id
            << ",\"selected_expert_ids\":[";
        for (std::int64_t expert = 0; expert < tensor->ne[0]; ++expert) {
            if (expert != 0) {
                state.output << ',';
            }
            state.output << expert_ids[static_cast<std::size_t>(
                token_column * tensor->ne[0] + expert)];
        }
        const auto observed_at_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        state.output
            << "],\"selected_expert_weights\":null"
            << ",\"observed_at_ns\":" << observed_at_ns << "}\n";
    }

    if (!state.output) {
        state.error = "failed to write router trace";
        return false;
    }
    return true;
}

bool empty_trace_callback(ggml_tensor *, bool, void *) {
    return false;
}

bool write_token_sequence(
    const std::string & path,
    const std::vector<llama_token> & prompt_tokens,
    const std::vector<llama_token> & generated_tokens) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        return false;
    }
    const auto write_ids = [&output](const std::vector<llama_token> & tokens) {
        output << '[';
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            if (index != 0) {
                output << ',';
            }
            output << tokens[index];
        }
        output << ']';
    };
    output << "{\n  \"schema_version\": \"" << schema_version << "\",\n";
    output << "  \"prompt_token_ids\": ";
    write_ids(prompt_tokens);
    output << ",\n  \"generated_token_ids\": ";
    write_ids(generated_tokens);
    output << "\n}\n";
    return static_cast<bool>(output);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::printf(
            "expertflow-router-probe schema %s, llama.cpp %s (%s)\n",
            schema_version,
            llama_release,
            llama_revision);
        return 0;
    }
    options config;
    if (!parse_options(argc, argv, config)) {
        usage(argv[0]);
        return 2;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "unable to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(
        vocab, config.prompt.c_str(), config.prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "unable to determine prompt token count\n");
        llama_model_free(model);
        return 1;
    }
    std::vector<llama_token> prompt_tokens(static_cast<std::size_t>(n_prompt));
    if (llama_tokenize(
            vocab,
            config.prompt.c_str(),
            config.prompt.size(),
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size()),
            true,
            true) < 0) {
        std::fprintf(stderr, "unable to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    const bool full_trace = config.mode == trace_mode::full;
    trace_state trace(
        config.trace_path,
        full_trace,
        config.capture_tensor,
        config.capture_token_index,
        config.capture_output);
    if (full_trace && !trace.output) {
        std::fprintf(stderr, "unable to open trace output\n");
        llama_model_free(model);
        return 1;
    }
    if (trace.capture_enabled && !trace.capture_output) {
        std::fprintf(stderr, "unable to open capture output\n");
        llama_model_free(model);
        return 1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<std::uint32_t>(n_prompt + config.n_predict + 8);
    context_params.n_batch = 1;
    context_params.n_ubatch = 1;
    context_params.no_perf = false;
    if (full_trace) {
        context_params.cb_eval = router_trace_callback;
        context_params.cb_eval_user_data = &trace;
    } else if (config.mode == trace_mode::empty) {
        context_params.cb_eval = empty_trace_callback;
    }

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "unable to create context\n");
        llama_model_free(model);
        return 1;
    }
    llama_set_n_threads(context, config.threads, config.threads);

    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::vector<llama_token> generated_tokens;
    int exit_code = 0;

    const auto decode_token = [&](
        llama_token & token, const char * phase, std::uint64_t token_index) {
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (full_trace) {
            trace.begin_forward(batch, phase, token_index);
        }
        const int decode_result = llama_decode(context, batch);
        if (full_trace) {
            trace.end_forward();
        }
        if (decode_result != 0) {
            std::fprintf(stderr, "decode failed with code %d: %s\n", decode_result, trace.error.c_str());
            return false;
        }
        if (full_trace) {
            ++trace.forward_id;
        }
        return true;
    };

    for (std::size_t index = 0; index < prompt_tokens.size(); ++index) {
        if (!decode_token(prompt_tokens[index], "prefill", index)) {
            exit_code = 1;
            break;
        }
    }

    for (int generated = 0; exit_code == 0 && generated < config.n_predict; ++generated) {
        const llama_token token = llama_sampler_sample(sampler, context, -1);
        generated_tokens.push_back(token);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        if (generated + 1 < config.n_predict &&
            !decode_token(
                generated_tokens.back(),
                "decode",
                prompt_tokens.size() + static_cast<std::size_t>(generated))) {
            exit_code = 1;
            break;
        }
    }

    if (exit_code == 0 &&
        !write_token_sequence(config.tokens_path, prompt_tokens, generated_tokens)) {
        std::fprintf(stderr, "unable to write token sequence\n");
        exit_code = 1;
    }
    if (full_trace && trace.hook_order == 0) {
        std::fprintf(
            stderr,
            "trace produced zero events after %llu callback asks, %llu selected asks, and %llu observations; routing candidates:\n",
            static_cast<unsigned long long>(trace.callback_asks),
            static_cast<unsigned long long>(trace.selected_asks),
            static_cast<unsigned long long>(trace.callback_observations));
        std::fprintf(stderr, "last callback error: %s\n", trace.error.c_str());
        for (const std::string & name : trace.observed_tensor_names) {
            std::fprintf(stderr, "  %s\n", name.c_str());
        }
        std::fprintf(stderr, "observation tensor names:\n");
        for (const std::string & name : trace.observation_tensor_names) {
            std::fprintf(stderr, "  %s\n", name.c_str());
        }
        exit_code = 1;
    }
    if (trace.capture_enabled && !trace.capture_written) {
        std::fprintf(
            stderr,
            "capture produced zero values for tensor %s at token %llu: %s\n",
            trace.capture_tensor.c_str(),
            static_cast<unsigned long long>(trace.capture_token_index),
            trace.error.c_str());
        exit_code = 1;
    }

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    return exit_code;
}
