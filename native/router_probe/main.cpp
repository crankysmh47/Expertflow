#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "live_cache_config.h"

#include <chrono>
#include <array>
#include <clocale>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
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
    counter,
    metadata,
    boundary,
    ids,
    disabled,
};

struct options {
    std::string model_path;
    std::string trace_path;
    std::string tokens_path;
    std::string performance_path;
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
        "usage: %s -m MODEL --tokens FILE --performance FILE [--trace FILE | --no-trace] "
        "[--trace-mode full|empty|counter|metadata|boundary|ids|disabled] "
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
    if (std::strcmp(value, "counter") == 0) {
        output = trace_mode::counter;
        return true;
    }
    if (std::strcmp(value, "metadata") == 0) {
        output = trace_mode::metadata;
        return true;
    }
    if (std::strcmp(value, "boundary") == 0) {
        output = trace_mode::boundary;
        return true;
    }
    if (std::strcmp(value, "ids") == 0) {
        output = trace_mode::ids;
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
        } else if (std::strcmp(argument, "--performance") == 0 && index + 1 < argc) {
            result.performance_path = argv[++index];
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
           !result.performance_path.empty() &&
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

constexpr std::uint64_t metadata_canary_token = UINT64_C(0xd1a6c0decafef00d);
constexpr std::uint16_t metadata_canary_layer = UINT16_C(0xffff);

struct trace_metadata_event {
    std::uint64_t token_index = 0;
    std::uint16_t layer_id = 0;
};

struct trace_metadata_state {
    trace_metadata_event * events = nullptr;
    std::size_t capacity = 0;
    std::size_t count = 0;
    std::uint64_t current_token_index = 0;
    bool overflow = false;
};

constexpr std::int32_t expert_id_canary = INT32_C(0x5a17c0de);

struct trace_ids_state {
    trace_metadata_event * events = nullptr;
    std::int32_t * expert_ids = nullptr;
    std::size_t capacity = 0;
    std::size_t count = 0;
    std::uint64_t current_token_index = 0;
    bool overflow = false;
    bool invalid_tensor = false;
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

bool counter_trace_callback(ggml_tensor *, bool ask, void * user_data) {
    if (ask) {
        ++*static_cast<std::uint64_t *>(user_data);
    }
    return false;
}

bool metadata_trace_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    if (!ask) {
        return false;
    }
    const char * name = ggml_get_name(tensor);
    if (!starts_with(name, topk_prefix)) {
        return false;
    }
    auto & state = *static_cast<trace_metadata_state *>(user_data);
    const int layer_id = parse_layer_id(name);
    if (layer_id < 0 || state.count >= state.capacity) {
        state.overflow = true;
        return false;
    }
    state.events[state.count++] = {
        state.current_token_index,
        static_cast<std::uint16_t>(layer_id),
    };
    return false;
}

bool boundary_trace_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    const bool selected_experts = starts_with(ggml_get_name(tensor), topk_prefix);
    if (ask) {
        return selected_experts;
    }
    if (selected_experts) {
        ++*static_cast<std::uint64_t *>(user_data);
    }
    return true;
}

bool ids_trace_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    const char * name = ggml_get_name(tensor);
    const bool selected_experts = starts_with(name, topk_prefix);
    if (ask) {
        return selected_experts;
    }
    if (!selected_experts) {
        return true;
    }

    auto & state = *static_cast<trace_ids_state *>(user_data);
    const int layer_id = parse_layer_id(name);
    if (layer_id < 0 || state.count >= state.capacity) {
        state.overflow = true;
        return true;
    }
    if (!ggml_is_contiguous(tensor) || tensor->type != GGML_TYPE_I32 ||
        tensor->ne[0] != 8 || tensor->ne[1] != 1 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        state.invalid_tensor = true;
        return true;
    }

    ggml_backend_tensor_get(
        tensor,
        state.expert_ids + state.count * 8,
        0,
        8 * sizeof(std::int32_t));
    state.events[state.count++] = {
        state.current_token_index,
        static_cast<std::uint16_t>(layer_id),
    };
    return true;
}

std::string json_escape(const std::string & value) {
    std::ostringstream escaped;
    for (const unsigned char byte : value) {
        switch (byte) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (byte < 0x20) {
                    escaped << "\\u" << std::hex << std::setw(4)
                            << std::setfill('0') << static_cast<int>(byte)
                            << std::dec << std::setfill(' ');
                } else {
                    escaped << static_cast<char>(byte);
                }
        }
    }
    return escaped.str();
}

bool write_token_sequence(
    const std::string & path,
    const llama_vocab * vocab,
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
    std::string generated_text;
    for (const llama_token token : generated_tokens) {
        std::vector<char> piece(32);
        int32_t length = llama_token_to_piece(
            vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
        if (length < 0) {
            piece.resize(static_cast<std::size_t>(-length));
            length = llama_token_to_piece(
                vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
        }
        if (length < 0) {
            return false;
        }
        generated_text.append(piece.data(), static_cast<std::size_t>(length));
    }

    output << "{\n  \"schema_version\": \"" << schema_version << "\",\n";
    output << "  \"prompt_token_ids\": ";
    write_ids(prompt_tokens);
    output << ",\n  \"generated_token_ids\": ";
    write_ids(generated_tokens);
    output << ",\n  \"generated_text\": \"" << json_escape(generated_text) << "\"";
    output << "\n}\n";
    return static_cast<bool>(output);
}

bool write_performance(
    const std::string & path,
    const llama_perf_context_data & perf,
    std::size_t prompt_tokens,
    std::size_t generated_tokens,
    double prompt_eval_ms,
    double decode_eval_ms,
    double end_to_end_ms,
    double time_to_first_token_ms,
    const std::vector<double> & decode_token_latencies_ms) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        return false;
    }
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"schema_version\":\"1.0.0\""
        << ",\"measurement_kind\":\"measured_host_wall_and_llama_counters\""
        << ",\"prompt_tokens\":" << prompt_tokens
        << ",\"generated_tokens\":" << generated_tokens
        << ",\"prompt_eval_ms\":" << prompt_eval_ms
        << ",\"decode_eval_ms\":" << decode_eval_ms
        << ",\"end_to_end_ms\":" << end_to_end_ms
        << ",\"time_to_first_token_ms\":" << time_to_first_token_ms
        << ",\"llama_t_p_eval_ms\":" << perf.t_p_eval_ms
        << ",\"llama_t_eval_ms\":" << perf.t_eval_ms
        << ",\"llama_n_p_eval\":" << perf.n_p_eval
        << ",\"llama_n_eval\":" << perf.n_eval
        << ",\"decode_token_latencies_ms\":[";
    for (std::size_t index = 0; index < decode_token_latencies_ms.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << decode_token_latencies_ms[index];
    }
    output << "]}\n";
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

    const live_cache_config cache_config = live_cache_config_parse(
        [](const char * name) { return std::getenv(name); });
    if (!cache_config.valid) {
        std::fprintf(stderr, "invalid live-cache configuration: %s\n", cache_config.error.c_str());
        return 3;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    std::array<llama_model_tensor_buft_override, 4> tensor_overrides = {};
    if (cache_config.use_tensor_overrides()) {
        const auto patterns = live_cache_tensor_override_patterns();
        for (std::size_t index = 0; index < patterns.size(); ++index) {
            tensor_overrides[index] = {
                patterns[index],
                ggml_backend_cpu_buffer_type(),
            };
        }
        model_params.tensor_buft_overrides = tensor_overrides.data();
    }
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

    const std::size_t metadata_capacity =
        static_cast<std::size_t>(n_prompt + config.n_predict + 1) * 30;
    std::vector<trace_metadata_event> metadata_storage;
    trace_metadata_state metadata;
    if (config.mode == trace_mode::metadata) {
        metadata_storage.resize(metadata_capacity + 2);
        metadata_storage.front() = {metadata_canary_token, metadata_canary_layer};
        metadata_storage.back() = {metadata_canary_token, metadata_canary_layer};
        metadata.events = metadata_storage.data() + 1;
        metadata.capacity = metadata_capacity;
    }
    std::vector<trace_metadata_event> ids_event_storage;
    std::vector<std::int32_t> ids_value_storage;
    trace_ids_state ids;
    if (config.mode == trace_mode::ids) {
        ids_event_storage.resize(metadata_capacity + 2);
        ids_event_storage.front() = {metadata_canary_token, metadata_canary_layer};
        ids_event_storage.back() = {metadata_canary_token, metadata_canary_layer};
        ids_value_storage.resize(metadata_capacity * 8 + 2);
        ids_value_storage.front() = expert_id_canary;
        ids_value_storage.back() = expert_id_canary;
        ids.events = ids_event_storage.data() + 1;
        ids.expert_ids = ids_value_storage.data() + 1;
        ids.capacity = metadata_capacity;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<std::uint32_t>(n_prompt + config.n_predict + 8);
    context_params.n_batch = 1;
    context_params.n_ubatch = 1;
    context_params.no_perf = false;
    std::uint64_t callback_counter = 0;
    std::uint64_t boundary_counter = 0;
    if (full_trace) {
        context_params.cb_eval = router_trace_callback;
        context_params.cb_eval_user_data = &trace;
    } else if (config.mode == trace_mode::empty) {
        context_params.cb_eval = empty_trace_callback;
    } else if (config.mode == trace_mode::counter) {
        context_params.cb_eval = counter_trace_callback;
        context_params.cb_eval_user_data = &callback_counter;
    } else if (config.mode == trace_mode::metadata) {
        context_params.cb_eval = metadata_trace_callback;
        context_params.cb_eval_user_data = &metadata;
    } else if (config.mode == trace_mode::boundary) {
        context_params.cb_eval = boundary_trace_callback;
        context_params.cb_eval_user_data = &boundary_counter;
    } else if (config.mode == trace_mode::ids) {
        context_params.cb_eval = ids_trace_callback;
        context_params.cb_eval_user_data = &ids;
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
    std::vector<double> decode_token_latencies_ms;
    int exit_code = 0;

    const auto decode_token = [&](
        llama_token & token, const char * phase, std::uint64_t token_index) {
        llama_batch batch = llama_batch_get_one(&token, 1);
        metadata.current_token_index = token_index;
        ids.current_token_index = token_index;
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

    const auto benchmark_started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < prompt_tokens.size(); ++index) {
        if (!decode_token(prompt_tokens[index], "prefill", index)) {
            exit_code = 1;
            break;
        }
    }
    const auto prompt_finished = std::chrono::steady_clock::now();
    auto first_token_sampled = prompt_finished;
    auto previous_token_sampled = prompt_finished;

    for (int generated = 0; exit_code == 0 && generated < config.n_predict; ++generated) {
        const llama_token token = llama_sampler_sample(sampler, context, -1);
        const auto token_sampled = std::chrono::steady_clock::now();
        if (generated == 0) {
            first_token_sampled = token_sampled;
        } else {
            decode_token_latencies_ms.push_back(
                std::chrono::duration<double, std::milli>(
                    token_sampled - previous_token_sampled).count());
        }
        previous_token_sampled = token_sampled;
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
    const auto benchmark_finished = std::chrono::steady_clock::now();

    if (exit_code == 0 &&
        !write_token_sequence(config.tokens_path, vocab, prompt_tokens, generated_tokens)) {
        std::fprintf(stderr, "unable to write token sequence\n");
        exit_code = 1;
    }
    if (exit_code == 0) {
        const llama_perf_context_data perf = llama_perf_context(context);
        const double prompt_eval_ms = std::chrono::duration<double, std::milli>(
            prompt_finished - benchmark_started).count();
        const double decode_eval_ms = std::chrono::duration<double, std::milli>(
            benchmark_finished - prompt_finished).count();
        const double end_to_end_ms = std::chrono::duration<double, std::milli>(
            benchmark_finished - benchmark_started).count();
        const double time_to_first_token_ms = std::chrono::duration<double, std::milli>(
            first_token_sampled - benchmark_started).count();
        if (!write_performance(
                config.performance_path,
                perf,
                prompt_tokens.size(),
                generated_tokens.size(),
                prompt_eval_ms,
                decode_eval_ms,
                end_to_end_ms,
                time_to_first_token_ms,
                decode_token_latencies_ms)) {
            std::fprintf(stderr, "unable to write performance result\n");
            exit_code = 1;
        }
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
    if (config.mode == trace_mode::counter) {
        std::fprintf(
            stderr,
            "trace_counter=%llu\n",
            static_cast<unsigned long long>(callback_counter));
    }
    if (config.mode == trace_mode::boundary) {
        std::fprintf(
            stderr,
            "trace_boundary_count=%llu\n",
            static_cast<unsigned long long>(boundary_counter));
        if (boundary_counter == 0) {
            exit_code = 1;
        }
    }
    if (config.mode == trace_mode::metadata) {
        const bool canaries_valid =
            metadata_storage.front().token_index == metadata_canary_token &&
            metadata_storage.front().layer_id == metadata_canary_layer &&
            metadata_storage.back().token_index == metadata_canary_token &&
            metadata_storage.back().layer_id == metadata_canary_layer;
        std::fprintf(
            stderr,
            "trace_metadata_count=%llu trace_metadata_capacity=%llu "
            "trace_metadata_overflow=%d trace_metadata_canaries=%d\n",
            static_cast<unsigned long long>(metadata.count),
            static_cast<unsigned long long>(metadata.capacity),
            metadata.overflow ? 1 : 0,
            canaries_valid ? 1 : 0);
        if (metadata.count == 0 || metadata.overflow || !canaries_valid) {
            exit_code = 1;
        }
    }
    if (config.mode == trace_mode::ids) {
        const bool event_canaries_valid =
            ids_event_storage.front().token_index == metadata_canary_token &&
            ids_event_storage.front().layer_id == metadata_canary_layer &&
            ids_event_storage.back().token_index == metadata_canary_token &&
            ids_event_storage.back().layer_id == metadata_canary_layer;
        const bool id_canaries_valid =
            ids_value_storage.front() == expert_id_canary &&
            ids_value_storage.back() == expert_id_canary;
        bool expert_ids_valid = true;
        for (std::size_t index = 0; index < ids.count * 8; ++index) {
            if (ids.expert_ids[index] < 0 || ids.expert_ids[index] >= 128) {
                expert_ids_valid = false;
                break;
            }
        }
        std::fprintf(
            stderr,
            "trace_ids_count=%llu trace_ids_capacity=%llu "
            "trace_ids_overflow=%d trace_ids_invalid_tensor=%d "
            "trace_ids_canaries=%d trace_ids_values_valid=%d\n",
            static_cast<unsigned long long>(ids.count),
            static_cast<unsigned long long>(ids.capacity),
            ids.overflow ? 1 : 0,
            ids.invalid_tensor ? 1 : 0,
            event_canaries_valid && id_canaries_valid ? 1 : 0,
            expert_ids_valid ? 1 : 0);
        if (ids.count == 0 || ids.overflow || ids.invalid_tensor ||
            !event_canaries_valid || !id_canaries_valid || !expert_ids_valid) {
            exit_code = 1;
        }
    }

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    return exit_code;
}
