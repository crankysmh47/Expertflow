from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_temporal_layer24_predictor import (
    _choose_validation_candidate,
    main,
)


def test_test_refuses_to_open_without_selection_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selection lock"):
        main(["test", "--manifest", str(tmp_path / "manifest.json"), "--output", str(tmp_path)])


def test_validation_selection_uses_fixed_tie_breaks() -> None:
    candidates = [
        {
            "policy": "t0.2_transition",
            "candidate_width": 8,
            "weights": None,
            "latency_batch1_cpu": {"p95_ns": 100},
            "metrics": {"recall_at_8": 0.5},
            "shadow": {
                "ready_improvement_over_reactive": 10,
                "eviction_regret": 2,
                "wasted_predicted_bytes": 100,
            },
        },
        {
            "policy": "t0.3_combined",
            "candidate_width": 8,
            "weights": [0.5, 0.25, 0.25],
            "latency_batch1_cpu": {"p95_ns": 100},
            "metrics": {"recall_at_8": 0.5},
            "shadow": {
                "ready_improvement_over_reactive": 10,
                "eviction_regret": 2,
                "wasted_predicted_bytes": 100,
            },
        },
    ]
    assert _choose_validation_candidate(candidates)["policy"] == "t0.2_transition"


def test_second_test_open_fails_without_modifying_metrics(tmp_path: Path) -> None:
    lock = tmp_path / "selection-lock.json"
    metrics = tmp_path / "test-metrics.json"
    lock.write_text(json.dumps({"test_opened": True}), encoding="utf-8")
    metrics.write_text('{"sentinel":true}\n', encoding="utf-8")
    before = metrics.read_bytes()
    with pytest.raises(ValueError, match="already been evaluated"):
        main(["test", "--manifest", str(tmp_path / "manifest.json"), "--output", str(tmp_path)])
    assert metrics.read_bytes() == before

