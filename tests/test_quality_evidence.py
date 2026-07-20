from __future__ import annotations

import json
from pathlib import Path

from expertflow.quality.manifest import canonical_manifest_hash


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/evidence/q6-quality-preserving/quality-manifest.json"
RESULTS = ROOT / "docs/evidence/q6-quality-preserving/q1-quality-results.json"


def test_quality_manifest_is_frozen_and_complete() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert len(data["datasets"]["wikitext"]["revision"]) == 40
    assert len(data["datasets"]["mmlu"]["revision"]) == 40
    assert data["mmlu"]["item_count"] == 100
    assert data["wikitext"] == {"chunk_count": 4, "chunk_tokens": 2048, "token_count": 8192}
    assert data["thresholds"]["perplexity_relative_max"] == 0.005
    assert data["manifest_sha256"] == canonical_manifest_hash(data)


def test_quality_manifest_pins_runtime_identity() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert data["runtime"]["llama_upstream"] == "a7312ae94f801fc9c6786dc56e38df57b964f697"
    assert data["runtime"]["llama_patch_commit"] == "29857466d39cc532cefc1633ac14e521849541fe"
    assert data["runtime"]["feature_flag"] == "LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER=0"
    assert len(data["runtime_artifacts"]) >= 7
    for identity in (*data["executables"].values(), data["model"]):
        assert identity["bytes"] > 0
        assert len(identity["sha256"]) == 64


def test_q1_stop_is_derived_from_frozen_perplexity_gate() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    perplexity = result["perplexity"]

    expected = perplexity["candidate_final"] / perplexity["reference_final"] - 1.0
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert perplexity["relative_change"] == expected
    assert expected > manifest["thresholds"]["perplexity_relative_max"]
    assert perplexity["pass"] is False
    assert result["decision"] == "OPTION 1 Q1 STOP"
    assert result["claims"] == {
        "quality_preserving": False,
        "runtime_speedup": False,
        "q1_pass": False,
    }
