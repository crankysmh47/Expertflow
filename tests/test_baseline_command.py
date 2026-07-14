from pathlib import Path

from expertflow.runtime.baseline import BaselineRunConfig, build_llama_command


def test_builds_deterministic_single_turn_command() -> None:
    config = BaselineRunConfig(
        executable=Path(r"C:\runtime\llama-cli.exe"),
        model=Path(r"C:\models\gemma.gguf"),
        prompt="Why does MoE locality matter?",
        log_file=Path(r"C:\runs\llama.log"),
        gpu_layers="auto",
        context_size=1024,
        predict_tokens=32,
        threads=12,
    )

    command = build_llama_command(config)

    assert command == [
        r"C:\runtime\llama-cli.exe",
        "--model",
        r"C:\models\gemma.gguf",
        "--prompt",
        "Why does MoE locality matter?",
        "--seed",
        "42",
        "--temp",
        "0",
        "--ctx-size",
        "1024",
        "--predict",
        "32",
        "--gpu-layers",
        "auto",
        "--threads",
        "12",
        "--threads-batch",
        "12",
        "--conversation",
        "--single-turn",
        "--no-display-prompt",
        "--perf",
        "--log-file",
        r"C:\runs\llama.log",
    ]


def test_rejects_invalid_baseline_limits() -> None:
    try:
        BaselineRunConfig(
            executable=Path("llama-cli"),
            model=Path("model.gguf"),
            prompt="test",
            log_file=Path("llama.log"),
            gpu_layers="0",
            context_size=0,
            predict_tokens=1,
            threads=1,
        )
    except ValueError as error:
        assert "context_size" in str(error)
    else:
        raise AssertionError("invalid context size was accepted")
