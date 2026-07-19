import json
from pathlib import Path

import pytest

from expertflow.cli.main import main


ROOT = Path(__file__).parents[1]
EXPECTED_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20]


@pytest.mark.parametrize(
    ("name", "goal", "context", "slots"),
    [
        ("max-performance.json", "latency", 2048, 1),
        ("max-throughput.json", "throughput", 8192, 4),
        ("max-context.json", "context", 262144, 1),
        ("max-agentic.json", "agentic", 8192, 4),
    ],
)
def test_deployment_profiles_are_portable_and_measured(name: str, goal: str, context: int, slots: int) -> None:
    path = ROOT / "deployments" / name
    deployment = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    assert deployment["goal"] == goal
    assert deployment["context"] == context
    assert deployment["parallel_slots"] == slots
    assert deployment["placement"]["static_expert_layers"] == EXPECTED_LAYERS
    assert deployment["model"]["sha256"] == "089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba"
    assert deployment["runtime"]["llama_commit"] == "451224ab4d12a616dc3e16e8c8063f4b331f531c"
    assert deployment["measurement_status"] == "measured"
    assert deployment["quality"]["strict_ppl_gate_pass"] is False
    assert "Hank47" not in text
    assert "C:\\models" not in text


@pytest.mark.parametrize("goal", ["throughput", "context", "agentic"])
def test_optimize_uses_frozen_measured_profile(goal: str, tmp_path: Path, capsys) -> None:
    output = tmp_path / f"{goal}.json"
    assert main(["optimize", "D:/models/model.gguf", "--goal", goal, "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    deployment = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert deployment["goal"] == goal
    assert deployment["model"]["path"] == "D:\\models\\model.gguf" or deployment["model"]["path"] == "D:/models/model.gguf"


def test_agentic_examples_use_configurable_openai_base_url() -> None:
    client = (ROOT / "examples/openai_client.py").read_text(encoding="utf-8")
    agent = (ROOT / "examples/agentic_session.py").read_text(encoding="utf-8")
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "EXPERTFLOW_BASE_URL" in client
    assert "/chat/completions" in client
    assert '"max_tokens": 96' in client
    assert "reasoning_content" in client
    assert "ThreadPoolExecutor" in agent
    assert ' * 80' in agent
    assert "EXPERTFLOW_BASE_URL=http://127.0.0.1:8080/v1" in env
    start = (ROOT / "scripts/start_expertflow.ps1").read_text(encoding="utf-8")
    assert "EXPERTFLOW_LLAMA_SERVER" in start
    assert "-np" in start
    assert (ROOT / "scripts/stop_expertflow.ps1").is_file()
