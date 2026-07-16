#include "live_cache_config.h"

#include <array>
#include <cctype>
#include <cstring>

namespace {

constexpr std::array<const char *, 6> subordinate_names = {
    "EXPERTFLOW_LIVE_CACHE_MODE",
    "EXPERTFLOW_LIVE_CACHE_LAYER",
    "EXPERTFLOW_LIVE_CACHE_LAYERS",
    "EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE",
    "EXPERTFLOW_LIVE_CACHE_LOG",
    "EXPERTFLOW_LIVE_CACHE_FORCE_EVICT",
};

bool is_absolute_windows_path(const char * value) {
    return value != nullptr && std::strlen(value) >= 3 &&
        std::isalpha(static_cast<unsigned char>(value[0])) && value[1] == ':' &&
        (value[2] == '\\' || value[2] == '/');
}

live_cache_config invalid(const char * message) {
    live_cache_config result;
    result.error = message;
    return result;
}

bool parse_layer_list(const char * value, live_cache_config & result) {
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const char * cursor = value;
    int previous = -1;
    while (*cursor != '\0') {
        if (!std::isdigit(static_cast<unsigned char>(*cursor))) {
            return false;
        }
        int layer_id = 0;
        do {
            layer_id = layer_id * 10 + (*cursor - '0');
            if (layer_id >= 30) {
                return false;
            }
            ++cursor;
        } while (std::isdigit(static_cast<unsigned char>(*cursor)));
        if (layer_id <= previous || result.layer_count >= result.layer_ids.size()) {
            return false;
        }
        result.layer_ids[result.layer_count++] = layer_id;
        result.requested_layers[layer_id] = true;
        result.enabled_layers[layer_id] = true;
        previous = layer_id;
        if (*cursor == '\0') {
            break;
        }
        if (*cursor++ != ',' || *cursor == '\0') {
            return false;
        }
    }
    return result.layer_count > 0;
}

} // namespace

live_cache_config live_cache_config_parse(const live_cache_environment_getter & get_environment) {
    const char * enabled = get_environment("EXPERTFLOW_LIVE_CACHE");
    if (enabled == nullptr) {
        for (const char * name : subordinate_names) {
            if (get_environment(name) != nullptr) {
                return invalid("live-cache setting present while EXPERTFLOW_LIVE_CACHE is unset");
            }
        }
        live_cache_config result;
        result.valid = true;
        return result;
    }
    if (std::strcmp(enabled, "1") != 0) {
        return invalid("EXPERTFLOW_LIVE_CACHE must be exactly 1 when set");
    }

    const char * mode = get_environment("EXPERTFLOW_LIVE_CACHE_MODE");
    const char * layer = get_environment("EXPERTFLOW_LIVE_CACHE_LAYER");
    const char * layers = get_environment("EXPERTFLOW_LIVE_CACHE_LAYERS");
    const char * auto_eligible = get_environment("EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE");
    const int selection_count =
        (layer != nullptr ? 1 : 0) +
        (layers != nullptr ? 1 : 0) +
        (auto_eligible != nullptr ? 1 : 0);
    if (mode == nullptr || selection_count != 1) {
        return invalid("enabled live cache requires exactly one singular, plural, or auto-eligible layer setting");
    }

    live_cache_config result;
    if (auto_eligible != nullptr) {
        if (std::strcmp(auto_eligible, "1") != 0) {
            return invalid("EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE must be exactly 1 when set");
        }
        result.auto_eligible = true;
        result.layer_count = result.layer_ids.size();
        for (int layer_id = 0; layer_id < 30; ++layer_id) {
            result.layer_ids[layer_id] = layer_id;
            result.requested_layers[layer_id] = true;
            result.enabled_layers[layer_id] = true;
        }
    } else if (layer != nullptr) {
        if (std::strcmp(layer, "24") != 0) {
            return invalid("legacy singular live-cache layer must be exactly 24");
        }
        result.layer_id = 24;
        result.layer_ids[0] = 24;
        result.requested_layers[24] = true;
        result.enabled_layers[24] = true;
        result.layer_count = 1;
    } else if (!parse_layer_list(layers, result)) {
        return invalid("plural live-cache layers must be an ascending comma-separated decimal list from 0 to 29");
    }
    if (std::strcmp(mode, "passthrough") == 0) {
        result.mode = live_cache_mode::passthrough;
    } else if (std::strcmp(mode, "blocking") == 0) {
        result.mode = live_cache_mode::blocking;
    } else {
        return invalid("live-cache mode must be passthrough or blocking");
    }

    const char * log_path = get_environment("EXPERTFLOW_LIVE_CACHE_LOG");
    const char * force_evict = get_environment("EXPERTFLOW_LIVE_CACHE_FORCE_EVICT");
    if (result.mode == live_cache_mode::passthrough) {
        if (log_path != nullptr || force_evict != nullptr) {
            return invalid("passthrough mode rejects log and forced-eviction settings");
        }
    } else {
        if (!is_absolute_windows_path(log_path)) {
            return invalid("blocking mode requires an absolute Windows log path");
        }
        result.log_path = log_path;
        if (force_evict != nullptr) {
            if (std::strcmp(force_evict, "1") != 0) {
                return invalid("EXPERTFLOW_LIVE_CACHE_FORCE_EVICT must be exactly 1 when set");
            }
            result.force_evict = true;
        }
    }

    result.valid = true;
    return result;
}

live_cache_override_patterns live_cache_tensor_override_patterns(
        const live_cache_config & config) {
    return live_cache_tensor_override_patterns(config, 31);
}

live_cache_override_patterns live_cache_tensor_override_patterns(
        const live_cache_config & config,
        int n_gpu_layers) {
    live_cache_override_patterns patterns;
    int first_auto_layer = 0;
    if (config.auto_eligible) {
        int repeating_gpu_layers = n_gpu_layers > 0 ? n_gpu_layers - 1 : 0;
        if (repeating_gpu_layers > 30) {
            repeating_gpu_layers = 30;
        }
        first_auto_layer = 30 - repeating_gpu_layers;
    }
    for (std::size_t index = 0; index < config.layer_count; ++index) {
        const int layer_id = config.layer_ids[index];
        if (config.auto_eligible && layer_id < first_auto_layer) {
            continue;
        }
        const std::string prefix = "^blk\\." + std::to_string(layer_id) + "\\.";
        patterns.values[patterns.count++] = prefix + "ffn_gate_up_exps\\.weight$";
        patterns.values[patterns.count++] = prefix + "ffn_down_exps\\.weight$";
        patterns.values[patterns.count++] = prefix + "ffn_down_exps\\.scale$";
    }
    return patterns;
}
