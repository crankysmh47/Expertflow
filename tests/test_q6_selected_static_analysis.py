from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "q6_selected_analysis", Path("scripts/analyze_q6_selected_static.py")
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_summarize_uses_matched_pair_percentages() -> None:
    rows = [
        {"pair": 1, "mode": "off", "decode_tps": 10.0, "prompt_tps": 2.0, "wall_seconds": 4.0,
         "process_owned_peak_mib": 100.0, "valid": True},
        {"pair": 1, "mode": "on", "decode_tps": 11.0, "prompt_tps": 3.0, "wall_seconds": 3.5,
         "process_owned_peak_mib": 200.0, "valid": True},
        {"pair": 2, "mode": "on", "decode_tps": 18.0, "prompt_tps": 3.0, "wall_seconds": 3.5,
         "process_owned_peak_mib": 200.0, "valid": True},
        {"pair": 2, "mode": "off", "decode_tps": 20.0, "prompt_tps": 2.0, "wall_seconds": 4.0,
         "process_owned_peak_mib": 100.0, "valid": True},
    ]

    result = analysis.summarize(rows, bootstrap_samples=1000, seed=7)

    assert result["pair_count"] == 2
    assert result["off"]["decode_tps_mean"] == 15.0
    assert result["on"]["decode_tps_mean"] == 14.5
    assert result["paired_improvement_pct_mean"] == pytest.approx(0.0)
    assert result["paired_improvement_pct_values"] == pytest.approx([10.0, -10.0])


def test_summarize_supports_one_pair_as_rehearsal_without_inventing_dispersion() -> None:
    rows = [
        {"pair": 1, "mode": "off", "decode_tps": 20.0, "prompt_tps": 4.0,
         "wall_seconds": 8.0, "process_owned_peak_mib": 3000.0, "valid": True},
        {"pair": 1, "mode": "on", "decode_tps": 25.0, "prompt_tps": 3.0,
         "wall_seconds": 9.0, "process_owned_peak_mib": 11000.0, "valid": True},
    ]

    result = analysis.summarize(rows, bootstrap_samples=100, seed=7)

    assert result["pair_count"] == 1
    assert result["off"]["decode_tps_sample_sd"] is None
    assert result["on"]["decode_tps_sample_sd"] is None
    assert result["paired_improvement_pct_sample_sd"] is None
    assert result["paired_bootstrap_95_pct"] == [25.0, 25.0]
