import json

from expertflow.analysis.narrow_placement import compare_runs


def _write_run(tmp_path, name, generated, experts):
    prefix = tmp_path / name
    (tmp_path / f"{name}.tokens.json").write_text(
        json.dumps({"prompt_token_ids": [2, 3], "generated_token_ids": generated}),
        encoding="utf-8",
    )
    events = [
        {
            "phase": "prefill",
            "token_index": 0,
            "layer_id": layer,
            "selected_expert_ids": ids,
            "observed_at_ns": 100 + layer,
        }
        for layer, ids in enumerate(experts)
    ]
    (tmp_path / f"{name}.trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    return prefix


def test_compare_runs_reports_first_token_and_router_divergence(tmp_path):
    reference = _write_run(tmp_path, "reference", [10, 11, 12], [[1, 2], [3, 4]])
    candidate = _write_run(tmp_path, "candidate", [10, 99, 12], [[1, 2], [4, 3]])

    result = compare_runs(reference, candidate)

    assert result["prompt_token_parity"] is True
    assert result["generated_token_parity"] is False
    assert result["first_generated_divergence"] == {"index": 1, "reference": 11, "candidate": 99}
    assert result["router_parity"] is False
    assert result["first_router_divergence"]["layer_id"] == 1
    assert result["first_router_divergence"]["reference"] == [3, 4]
    assert result["first_router_divergence"]["candidate"] == [4, 3]


def test_compare_runs_ignores_observation_timestamp(tmp_path):
    reference = _write_run(tmp_path, "reference", [10], [[1, 2]])
    candidate = _write_run(tmp_path, "candidate", [10], [[1, 2]])
    event = json.loads((tmp_path / "candidate.trace.jsonl").read_text(encoding="utf-8"))
    event["observed_at_ns"] = 999999
    (tmp_path / "candidate.trace.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    result = compare_runs(reference, candidate)

    assert result["generated_token_parity"] is True
    assert result["router_parity"] is True
