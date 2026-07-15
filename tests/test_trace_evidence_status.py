import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "configs" / "trace-evidence-status.json"


def test_callback_derived_real_model_evidence_is_quarantined() -> None:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    assert status["schema_version"] == "1.0.0"
    assert status["trace_generation"]["label"] == "trace_v1_perturbing"
    assert status["trace_generation"]["corpus_collection_enabled"] is False
    assert status["trace_generation"]["eligible_for_final_claims"] is False

    canonical = status["canonical_trace_generation"]
    assert canonical["label"] == "trace_v2_canonical_segmented"
    assert canonical["runtime"] == "expertflow-canonical-observer-v1"
    assert canonical["corpus_collection_enabled"] is True
    assert canonical["eligible_for_final_claims"] is True
    assert canonical["smoke_decision"] == "accepted"

    roots = {item["id"]: item for item in status["real_model_roots"]}
    assert {
        "q4-baseline",
        "stratified-cuda",
        "stratified-vulkan",
        "heldout-vulkan",
        "physical-feasibility-vulkan",
        "gate3-clean-runtime",
        "gate3-divergence-audit",
    } <= roots.keys()
    assert all(item["excluded_from_final_claims"] for item in roots.values())

    claims = {item["id"]: item for item in status["derived_claims"]}
    assert {
        "locality",
        "static-policy",
        "lru-policy",
        "session-policy",
        "oracle-deadline",
        "static-96-93.28-percent",
    } <= claims.keys()
    assert all(item["excluded_from_final_claims"] for item in claims.values())
    assert claims["static-96-93.28-percent"]["status"] == "withdrawn_pending_recollection"

    fixture = status["offline_fixture"]
    assert fixture["path"] == "examples/replay/trace.jsonl"
    assert fixture["allowed_use"] == "synthetic_or_offline_validation_only"
    assert fixture["supports_real_model_policy_claims"] is False

    assert status["live_cache_enabled"] is False
    assert status["gate4_open"] is False
