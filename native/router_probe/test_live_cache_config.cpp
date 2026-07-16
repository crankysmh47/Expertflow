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

static void assert_layers(
        const live_cache_config & config,
        std::initializer_list<int> expected) {
    assert(config.layer_count == expected.size());
    std::size_t index = 0;
    for (const int layer_id : expected) {
        assert(config.layer_ids[index++] == layer_id);
        assert(config.enabled_layers[layer_id]);
    }
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

static void test_plural_layer_lists_are_exact_and_ascending() {
    const auto parse_layers = [](const char * layers) {
        return parse({
            { "EXPERTFLOW_LIVE_CACHE", "1" },
            { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
            { "EXPERTFLOW_LIVE_CACHE_LAYERS", layers },
            { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
        });
    };
    assert_layers(parse_layers("0,24"), { 0, 24 });
    assert_layers(parse_layers("0,7,14,21,29"), { 0, 7, 14, 21, 29 });

    std::string all_layers;
    for (int layer_id = 0; layer_id < 30; ++layer_id) {
        if (!all_layers.empty()) {
            all_layers += ",";
        }
        all_layers += std::to_string(layer_id);
    }
    const live_cache_config all = parse_layers(all_layers.c_str());
    assert(all.valid);
    assert(all.layer_count == 30);
    for (int layer_id = 0; layer_id < 30; ++layer_id) {
        assert(all.layer_ids[layer_id] == layer_id);
    }

    for (const char * invalid : {
            "", "24,0", "0,0", "-1,24", "0,30", "x,24", "0, 24", "0..29", "0,", ",24" }) {
        assert(!parse_layers(invalid).valid);
    }
    assert(!parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_LAYER", "24" },
        { "EXPERTFLOW_LIVE_CACHE_LAYERS", "0,24" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
    }).valid);
}

static void test_blocking_tensor_overrides_cover_every_configured_layer() {
    const live_cache_config config = parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_LAYERS", "0,24" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
    });
    const auto patterns = live_cache_tensor_override_patterns(config);
    assert(patterns.count == 6);
    assert(patterns.values[0] == "^blk\\.0\\.ffn_gate_up_exps\\.weight$");
    assert(patterns.values[1] == "^blk\\.0\\.ffn_down_exps\\.weight$");
    assert(patterns.values[2] == "^blk\\.0\\.ffn_down_exps\\.scale$");
    assert(patterns.values[3] == "^blk\\.24\\.ffn_gate_up_exps\\.weight$");
    assert(patterns.values[4] == "^blk\\.24\\.ffn_down_exps\\.weight$");
    assert(patterns.values[5] == "^blk\\.24\\.ffn_down_exps\\.scale$");
}

static void test_auto_eligible_mode_is_explicit_and_mutually_exclusive() {
    const live_cache_config automatic = parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
    });
    assert(automatic.valid);
    assert(automatic.auto_eligible);
    assert(automatic.layer_count == 30);
    for (int layer_id = 0; layer_id < 30; ++layer_id) {
        assert(automatic.layer_ids[layer_id] == layer_id);
        assert(automatic.requested_layers[layer_id]);
    }
    const auto ngl10 = live_cache_tensor_override_patterns(automatic, 10);
    assert(ngl10.count == 27);
    assert(ngl10.values[0] == "^blk\\.21\\.ffn_gate_up_exps\\.weight$");
    assert(ngl10.values[26] == "^blk\\.29\\.ffn_down_exps\\.scale$");

    assert(!parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_LAYERS", "21,24" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
    }).valid);
    assert(!parse({
        { "EXPERTFLOW_LIVE_CACHE", "1" },
        { "EXPERTFLOW_LIVE_CACHE_MODE", "blocking" },
        { "EXPERTFLOW_LIVE_CACHE_AUTO_ELIGIBLE", "0" },
        { "EXPERTFLOW_LIVE_CACHE_LOG", "C:\\runs\\cache.jsonl" },
    }).valid);
}

int main() {
    test_unset_environment_is_disabled();
    test_subordinate_variable_without_enable_fails();
    test_passthrough_initializes_without_tensor_override();
    test_blocking_requires_layer_24_and_absolute_log();
    test_force_evict_is_blocking_only();
    test_plural_layer_lists_are_exact_and_ascending();
    test_blocking_tensor_overrides_cover_every_configured_layer();
    test_auto_eligible_mode_is_explicit_and_mutually_exclusive();
    return 0;
}
