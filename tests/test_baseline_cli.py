from pathlib import Path

import expertflow.cli.main as cli


def test_baseline_cli_passes_explicit_run_contract(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "llama-cli.exe"
    model = tmp_path / "model.gguf"
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "run"
    executable.write_bytes(b"runtime")
    model.write_bytes(b"model")
    prompt.write_text("Explain MoE locality.", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(config, *, model_sha256, manifest_path):
        captured["config"] = config
        captured["model_sha256"] = model_sha256
        captured["manifest_path"] = manifest_path
        return {"return_code": 0}

    monkeypatch.setattr(cli, "run_measured_baseline", fake_run)

    result = cli.main(
        [
            "baseline",
            "--runtime",
            str(executable),
            "--model",
            str(model),
            "--model-sha256",
            "a" * 64,
            "--prompt-file",
            str(prompt),
            "--output-dir",
            str(output),
            "--gpu-layers",
            "0",
            "--ctx-size",
            "1024",
            "--predict",
            "8",
            "--threads",
            "6",
        ]
    )

    config = captured["config"]
    assert result == 0
    assert config.prompt == "Explain MoE locality."
    assert config.gpu_layers == "0"
    assert config.log_file == output.resolve() / "llama.log"
    assert captured["model_sha256"] == "a" * 64
    assert captured["manifest_path"] == output.resolve() / "manifest.json"
