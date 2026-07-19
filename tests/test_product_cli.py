import json
from pathlib import Path

from expertflow.cli.main import main
from expertflow.product.commands import build_runtime_command


ROOT = Path(__file__).parents[1]


def _last_json(capsys) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_demo_replay_is_model_free_and_verifies_hashes(capsys) -> None:
    assert main(["demo", "--replay"]) == 0
    report = _last_json(capsys)
    assert report["status"] == "pass"
    assert report["mode"] == "measured_evidence_replay"
    assert report["performance"]["expertflow_decode_tps"] == 28.13
    assert report["quality"]["strict_ppl_gate_pass"] is False
    assert report["cache_strategy"]["verdict"] == "NO CACHE OPPORTUNITY"
    assert report["evidence_hashes_verified"] is True


def test_compare_reports_frozen_stock_and_expertflow(capsys) -> None:
    deployment = ROOT / "docs/evidence/product-release/deployment-result.json"
    assert main(["compare", str(deployment)]) == 0
    report = _last_json(capsys)
    assert report["status"] == "pass"
    assert report["stock"]["decode_tps"] == 22.967
    assert report["expertflow"]["decode_tps"] == 28.13
    assert report["improvement_pct"] == 22.480080114947533
    assert report["quality"]["strict_ppl_gate_pass"] is False


def test_optimize_latency_writes_portable_deployment(tmp_path: Path, capsys) -> None:
    output = tmp_path / "deployment.json"
    assert main(["optimize", "model.gguf", "--goal", "latency", "--output", str(output)]) == 0
    report = _last_json(capsys)
    deployment = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert deployment["goal"] == "latency"
    assert deployment["model"]["path"] == "model.gguf"
    assert deployment["placement"]["static_expert_layers"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20]
    assert "Hank47" not in output.read_text(encoding="utf-8")


def test_profile_gguf_uses_committed_measured_profile(capsys) -> None:
    assert main(["profile", "model.gguf"]) == 0
    report = _last_json(capsys)
    assert report["status"] == "warning"
    assert report["measurement_kind"] == "measured_recorded_evidence_plus_live_path_inspection"
    assert report["routed_layer_count"] == 30
    assert report["expert_bank_bytes_per_layer"] == 685933056
    assert report["recommended_static_layers"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20]


def test_runtime_command_preserves_manifest_order_and_environment(tmp_path: Path) -> None:
    deployment = json.loads((ROOT / "docs/evidence/product-release/deployment-result.json").read_text(encoding="utf-8"))
    runtime = tmp_path / "llama-cli.exe"
    model = tmp_path / "model.gguf"
    runtime.write_bytes(b"runtime")
    model.write_bytes(b"model")
    command, environment = build_runtime_command(deployment, runtime=runtime, model=model)
    assert command[:3] == [str(runtime), "-m", str(model)]
    assert command[3:] == deployment["arguments"]
    assert environment["LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER"] == "0,1,2,3,4,5,6,7,8,9,15,20"


def test_unmeasured_profile_fails_instead_of_projecting(tmp_path: Path, capsys) -> None:
    output = tmp_path / "throughput.json"
    assert main(["optimize", "model.gguf", "--goal", "throughput", "--output", str(output)]) == 2
    report = _last_json(capsys)
    assert report["status"] == "failure"
    assert report["reason"] == "measured throughput profile is not available"
    assert not output.exists()
