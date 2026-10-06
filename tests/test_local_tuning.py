import copy
import json
import pytest


def test_unqualified_runtime_does_not_launch_tuning(tmp_path, monkeypatch):
    from expertflow.product.tuning import tune_profile
    profile = {"runtime":{"files":{},"version":"unknown"},"model":{"sha256":"0"*64},"host":{},"settings":{"threads":None},"status":"RUNNABLE-UNTUNED"}
    def forbidden(*a,**kw):
        pytest.fail("Unqualified model/runtime must launch zero processes")
    monkeypatch.setattr("expertflow.product.tuning.measure_profile", forbidden)
    result = tune_profile(profile, tmp_path / "job", budget_seconds=600)
    assert result["status"] == "UNSUPPORTED"
    assert result["model_processes"] == 0
    assert profile["settings"]["threads"] is None


def test_exact_token_mismatch_rejects_even_large_speedup():
    from expertflow.product.tuning import decide_pairs
    pairs = [{"baseline":{"tokens":[1,2],"decode_tps":10},"candidate":{"tokens":[1,3],"decode_tps":20}} for _ in range(5)]
    result = decide_pairs(pairs)
    assert result["status"] == "INCONCLUSIVE"
    assert "token" in result["reason"].lower()


def test_neutral_and_improved_outcomes_use_fixed_practical_gate():
    from expertflow.product.tuning import decide_pairs
    def pairs(tps):
        return [{"baseline":{"tokens":[1,2],"decode_tps":100},"candidate":{"tokens":[1,2],"decode_tps":tps}} for _ in range(5)]
    assert decide_pairs(pairs(101))["status"] == "NO-MEASURABLE-GAIN"
    result = decide_pairs(pairs(110))
    assert result["status"] == "VERIFIED-IMPROVEMENT"
    assert result["ci95_pct"][0] > 5


def test_noisy_confirmation_never_promotes_winner():
    from expertflow.product.tuning import decide_pairs
    pairs = [{"baseline":{"tokens":[1],"decode_tps":100},"candidate":{"tokens":[1],"decode_tps":t}} for t in [70,150,90,140,120]]
    assert decide_pairs(pairs)["status"] == "INCONCLUSIVE"


def test_candidate_failure_consumes_budget_and_retains_partial(tmp_path, monkeypatch):
    from expertflow.product.tuning import tune_profile
    profile={"runtime":{},"model":{},"host":{},"settings":{"threads":None},"status":"RUNNABLE-UNTUNED"}
    monkeypatch.setattr("expertflow.product.tuning.eligible_threads", lambda p: [8,12,16])
    def failing(*a,**kw):
        raise ValueError("OOM")
    monkeypatch.setattr("expertflow.product.tuning.measure_profile", failing)
    result=tune_profile(profile,tmp_path / "job",budget_seconds=600)
    assert result["status"] == "INCONCLUSIVE"
    assert result["model_attempts"] == 1
    assert result["model_processes"] == 0  # injected failure occurred before native launch
    assert "OOM" in result["reason"]
    assert (tmp_path / "job" / "registration.json").exists()
    assert (tmp_path / "job" / "report.json").exists()
    with pytest.raises(ValueError,match="exists|occupied"):
        tune_profile(profile,tmp_path / "job",budget_seconds=600)


def test_explicit_thread_baseline_cannot_be_promoted_as_defaults_gain(tmp_path,monkeypatch):
    from expertflow.product.tuning import tune_profile
    monkeypatch.setattr("expertflow.product.tuning.eligible_threads",lambda p:[8,12,16])
    monkeypatch.setattr("expertflow.product.tuning.measure_profile",lambda *a,**k:pytest.fail("Must not compare against an artificially poor thread override"))
    result=tune_profile({"settings":{"threads":1}},tmp_path / "job",budget_seconds=600)
    assert result["status"] == "UNSUPPORTED"
    assert result["model_processes"] == 0
    assert "default" in result["reason"]
