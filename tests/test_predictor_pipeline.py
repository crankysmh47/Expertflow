from pathlib import Path
import json

import pytest

from scripts.run_next_layer_predictor import _choose_expanded, _deduplication_provenance, main


def test_test_command_refuses_to_open_test_without_selection_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selection lock"):
        main([
            "test",
            "--manifest", str(tmp_path / "manifest.json"),
            "--output", str(tmp_path),
        ])


def _candidate(*, gain: int, regret: int, waste: int, recall: float) -> dict[str, object]:
    return {
        "metrics": {"recall_at_8": recall},
        "policy_shadows": {
            "observed_support": {
                "8": {
                    "ready_improvement_over_reactive": gain,
                    "eviction_regret": regret,
                    "wasted_predicted_bytes": waste,
                }
            },
            "all_ranked": {
                "8": {
                    "ready_improvement_over_reactive": gain,
                    "eviction_regret": regret,
                    "wasted_predicted_bytes": waste + 1,
                }
            },
        },
    }


def test_expanded_selection_uses_fixed_validation_tie_breaks() -> None:
    validation = {
        "b2_raw_count_pooled": _candidate(gain=10, regret=2, waste=100, recall=0.4),
        "b2_raw_count_separate": _candidate(gain=10, regret=2, waste=100, recall=0.4),
        "b2_source_normalized_pooled": _candidate(gain=10, regret=2, waste=100, recall=0.4),
        "b2_source_normalized_separate": _candidate(gain=10, regret=2, waste=100, recall=0.4),
    }

    selected = _choose_expanded(validation, widths=(8,))

    assert selected == {
        "model": "b2_raw_count_pooled",
        "candidate_width": 8,
        "admission_rule": "observed_support",
    }


def test_expanded_test_refuses_second_open_without_modifying_metrics(tmp_path: Path) -> None:
    lock = tmp_path / "selection-lock.json"
    metrics = tmp_path / "test-metrics.json"
    lock.write_text(json.dumps({"test_opened": True}), encoding="utf-8")
    metrics.write_text('{"sentinel": true}\n', encoding="utf-8")
    before = metrics.read_bytes()

    with pytest.raises(ValueError, match="already been evaluated"):
        main([
            "test",
            "--manifest", str(tmp_path / "manifest.json"),
            "--output", str(tmp_path),
            "--expanded",
        ])

    assert metrics.read_bytes() == before


def test_deduplication_provenance_is_loaded_from_hashed_frozen_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({
        "deduplication_policy": {
            "exact": "unique SHA-256",
            "normalized": "unique normalized SHA-256",
        }
    }), encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()

    provenance = _deduplication_provenance({
        "frozen_corpus": {"path": str(corpus), "bytes": corpus.stat().st_size, "sha256": digest}
    })

    assert provenance["frozen_corpus"]["sha256"] == digest
    assert provenance["policy"]["exact"] == "unique SHA-256"
