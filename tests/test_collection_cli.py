from pathlib import Path

import expertflow.cli.main as cli


def test_collect_pairs_cli_passes_pinned_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}", encoding="utf-8")
    probe = tmp_path / "probe.exe"
    probe.write_bytes(b"probe")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    output = tmp_path / "runs"
    captured: dict[str, object] = {}

    def fake_collect(corpus_path, config):
        captured["corpus"] = corpus_path
        captured["config"] = config
        return {
            "summary": {
                "conversation_count": 40,
                "passed": 40,
                "failed": 0,
                "skipped_valid": 0,
            }
        }

    monkeypatch.setattr(cli, "collect_trace_pairs", fake_collect)

    result = cli.main(
        [
            "collect-pairs",
            "--corpus",
            str(corpus),
            "--probe",
            str(probe),
            "--model",
            str(model),
            "--model-sha256",
            "a" * 64,
            "--output-dir",
            str(output),
            "--predict",
            "64",
            "--gpu-layers",
            "10",
            "--threads",
            "12",
        ]
    )

    assert result == 0
    assert captured["corpus"] == corpus.resolve()
    config = captured["config"]
    assert config.probe == probe.resolve()
    assert config.model == model.resolve()
    assert config.output_dir == output.resolve()
    assert config.n_predict == 64
    assert config.gpu_layers == 10
    assert config.threads == 12
