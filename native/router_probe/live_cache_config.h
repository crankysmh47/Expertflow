#pragma once

#include <functional>
#include <array>
#include <string>

enum class live_cache_mode {
    disabled,
    passthrough,
    blocking,
};

struct live_cache_config {
    bool valid = false;
    live_cache_mode mode = live_cache_mode::disabled;
    int layer_id = -1;
    bool auto_eligible = false;
    std::array<bool, 30> requested_layers = {};
    std::array<bool, 30> enabled_layers = {};
    std::array<int, 30> layer_ids = {};
    std::size_t layer_count = 0;
    bool force_evict = false;
    std::string log_path;
    std::string error;

    bool use_tensor_overrides() const {
        return valid && mode == live_cache_mode::blocking;
    }
};

struct live_cache_override_patterns {
    std::array<std::string, 90> values;
    std::size_t count = 0;
};

using live_cache_environment_getter = std::function<const char *(const char *)>;

live_cache_config live_cache_config_parse(const live_cache_environment_getter & get_environment);

live_cache_override_patterns live_cache_tensor_override_patterns(
    const live_cache_config & config);

live_cache_override_patterns live_cache_tensor_override_patterns(
    const live_cache_config & config,
    int n_gpu_layers);
