#include "live_cache_config.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <map>
#include <string>

static live_cache_config parse(const std::map<std::string, std::string> & values) {
    return live_cache_config_parse([&values](const char * name) -> const char * {
        const auto found = values.find(name);
        return found == values.end() ? nullptr : found->second.c_str();
    });
}

static void test_unset_environment_is_disabled() {
    const live_cache_config config = parse({});
    assert(config.valid);
    assert(config.mode == live_cache_mode::disabled);
    assert(!config.use_tensor_overrides());
}

static void test_subordinate_variable_without_enable_fails() {
    const live_cache_config config = parse({
        { "EXPERTFLOW_LIVE_CACHE_LAYER", "24" },
    });
    assert(!config.valid);
}

static void test_passthrough_initializes_without_tensor_override() {
    const live_cache_config config = parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "passthrough" },
        { "EXPERTFLOW_LIVE_CACHE_LAYER", "24" },
    });
    assert(config.valid);
    assert(config.mode == live_cache_mode::passthrough);
    assert(config.layer_id == 24);
    assert(!config.use_tensor_overrides());
}

static void test_blocking_requires_layer_24_and_absolute_log() {
    assert(!parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_LAYER", "23" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
    }).valid);
    assert(!parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_LAYER", "24" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "relative.jsonl" },
    }).valid);

    const live_cache_config config = parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_LAYER", "24" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
    });
    assert(config.valid);
    assert(config.mode == live_cache_mode::blocking);
    assert(config.use_tensor_overrides());
}

static void test_force_evict_is_blocking_only() {
    assert(!parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "passthrough" },
        { "EXPERTFLOW_LIVE_CACHE_LAYER", "24" },
        { "EXPERTFLOW_LIVE_CACHE_FORCE_EVICT", "1" },
    }).valid);

    const live_cache_config config = parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_LAYER", "24" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
        { "EXPERTFLOW_LIVE_CACHE_FORCE_EVICT", "1" },
    });
    assert(config.valid);
    assert(config.force_evict);
}

static void test_blocking_tensor_overrides_are_exactly_layer_24_experts() {
    const auto patterns = live_cache_tensor_override_patterns();
    assert(std::string(patterns[0]) == "^blk\\.24\\.ffn_gate_up_exps\\.weight$");
    assert(std::string(patterns[1]) == "^blk\\.24\\.ffn_down_exps\\.weight$");
    assert(std::string(patterns[2]) == "^blk\\.24\\.ffn_down_exps\\.scale$");
}

int main() {
    test_unset_environment_is_disabled();
    test_subordinate_variable_without_enable_fails();
    test_passthrough_initializes_without_tensor_override();
    test_blocking_requires_layer_24_and_absolute_log();
    test_force_evict_is_blocking_only();
    test_blocking_tensor_overrides_are_exactly_layer_24_experts();
    return 0;
}
