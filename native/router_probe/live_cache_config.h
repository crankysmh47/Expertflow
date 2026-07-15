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
    bool force_evict = false;
    std::string log_path;
    std::string error;

    bool use_tensor_overrides() const {
        return valid && mode == live_cache_mode::blocking;
    }
};

using live_cache_environment_getter = std::function<const char *(const char *)>;

live_cache_config live_cache_config_parse(const live_cache_environment_getter & get_environment);

std::array<const char *, 3> live_cache_tensor_override_patterns();
