import json

import pytest

from expertflow.trace.schema import TraceValidationError, parse_router_event


def valid_record() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "request_id": "req-001",
        "conversation_id": "conv-001",
        "turn_index": 0,
        "phase": "decode",
        "forward_id": 12,
        "hook_order": 41,
        "token_index": 20,
        "token_id": 42,
        "layer_id": 7,
        "selected_expert_ids": [2, 9],
        "selected_expert_weights": [0.75, 0.25],
        "observed_at_ns": 123_456_789,
    }


def test_parses_valid_router_event() -> None:
    event = parse_router_event(json.dumps(valid_record()), record_number=3)

    assert event.request_id == "req-001"
    assert event.phase == "decode"
    assert event.selected_expert_ids == (2, 9)
    assert event.selected_expert_weights == (0.75, 0.25)


def test_rejects_unknown_schema_with_record_number() -> None:
    record = valid_record()
    record["schema_version"] = "2.0.0"

    with pytest.raises(TraceValidationError, match=r"record 7:.*schema_version"):
        parse_router_event(json.dumps(record), record_number=7)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("phase", "training", "phase"),
        ("selected_expert_ids", [], "selected_expert_ids"),
        ("selected_expert_ids", [2, 2], "unique"),
        ("selected_expert_weights", [1.0], "same length"),
        ("layer_id", -1, "layer_id"),
    ],
)
def test_rejects_malformed_router_event(
    field: str, value: object, reason: str
) -> None:
    record = valid_record()
    record[field] = value

    with pytest.raises(TraceValidationError, match=rf"record 11:.*{reason}"):
        parse_router_event(json.dumps(record), record_number=11)


def test_rejects_invalid_json_with_record_number() -> None:
    with pytest.raises(TraceValidationError, match=r"record 5: invalid JSON"):
        parse_router_event("{", record_number=5)
