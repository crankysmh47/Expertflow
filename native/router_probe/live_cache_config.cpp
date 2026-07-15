#include "live_cache_config.h"

#include <array>
#include <cctype>
#include <cstring>

namespace {

constexpr std::array<const char *, 4> subordinate_names = {
    "EXPERTFLOW_LIVE_CACHE_MODE",
    "EXPERTFLOW_LIVE_CACHE_LAYER",
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
    if (mode == nullptr || layer == nullptr || std::strcmp(layer, "24") != 0) {
        return invalid("enabled live cache requires mode and layer 24");
    }

    live_cache_config result;
    result.layer_id = 24;
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

std::array<const char *, 3> live_cache_tensor_override_patterns() {
    return {
        "^blk\\.24\\.ffn_gate_up_exps\\.weight$",
        "^blk\\.24\\.ffn_down_exps\\.weight$",
        "^blk\\.24\\.ffn_down_exps\\.scale$",
    };
}
