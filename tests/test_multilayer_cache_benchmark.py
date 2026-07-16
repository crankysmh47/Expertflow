import json

import pytest

from expertflow.benchmark.multilayer import reconcile_multilayer_events


def _event(token, layer, selected, hits, misses):
    return {
        "schema_version": "1.2.0",
        "token_index": token,
        "layer_id": layer,
        "layer_access_sequence": token + 1,
        "selected": selected,
        "physical_slots": list(range(8)),
        "hits": hits,
        "blocking_misses": misses,
        "loads": [],
        "bytes_transferred": misses * 3_345_412,
        "blocking_duration_us": misses * 100,
        "final_resident_mapping": [],
    }


def test_reconciles_two_layer_events_and_aggregate_accounting(tmp_path):
    cache = tmp_path / "cache.jsonl"
    trace = tmp_path / "trace.jsonl"
    events = [
        _event(0, 0, list(range(8)), 0, 8),
        _event(0, 24, list(range(8, 16)), 2, 6),
        _event(1, 0, list(range(1, 9)), 7, 1),
        _event(1, 24, list(range(9, 17)), 6, 2),
    ]
    cache.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "forward_id": token,
                    "token_index": token,
                    "layer_id": layer,
                    "selected_expert_ids": selected,
                }
            )
            for token, layer, selected in (
                (0, 0, list(range(8))),
                (0, 24, list(range(8, 16))),
                (1, 0, list(range(1, 9))),
                (1, 24, list(range(9, 17))),
            )
        ),
        encoding="utf-8",
    )
    result = reconcile_multilayer_events(
        cache,
        trace,
        enabled_layers=(0, 24),
        arena_layout={
            "total_bytes": 214_107_392,
            "layers": {
                "0": {"gate_up_offset": 0, "end_offset": 107_053_696},
                "24": {"gate_up_offset": 107_053_696, "end_offset": 214_107_392},
            },
        },
    )
    assert result["events"] == 4
    assert result["expert_demands"] == 32
    assert result["hits"] == 15
    assert result["misses"] == 17
    assert result["bytes_transferred"] == 17 * 3_345_412
    assert result["layers"]["0"]["events"] == 2
    assert result["layers"]["24"]["misses"] == 8


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "out_of_order"])
def test_rejects_missing_extra_duplicate_or_out_of_order_layers(tmp_path, mutation):
    cache = tmp_path / "cache.jsonl"
    trace = tmp_path / "trace.jsonl"
    events = [
        _event(0, 0, list(range(8)), 0, 8),
        _event(0, 24, list(range(8, 16)), 0, 8),
    ]
    if mutation == "missing":
        events.pop()
    elif mutation == "extra":
        events.append(_event(0, 7, list(range(16, 24)), 0, 8))
    elif mutation == "duplicate":
        events[1]["layer_id"] = 0
    else:
        events.reverse()
    cache.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    trace.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        reconcile_multilayer_events(cache, trace, enabled_layers=(0, 24))
