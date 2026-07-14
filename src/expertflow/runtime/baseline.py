"""Deterministic llama.cpp baseline command contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BaselineRunConfig:
    """All behavior-changing inputs for one unmodified baseline run."""

    executable: Path
    model: Path
    prompt: str
    log_file: Path
    gpu_layers: str
    context_size: int
    predict_tokens: int
    threads: int
    seed: int = 42
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if self.predict_tokens <= 0:
            raise ValueError("predict_tokens must be positive")
        if self.threads <= 0:
            raise ValueError("threads must be positive")
        if self.gpu_layers not in {"auto", "all"}:
            try:
                numeric_layers = int(self.gpu_layers)
            except ValueError as error:
                raise ValueError(
                    "gpu_layers must be a non-negative integer, 'auto', or 'all'"
                ) from error
            if numeric_layers < 0:
                raise ValueError("gpu_layers must not be negative")


def build_llama_command(config: BaselineRunConfig) -> list[str]:
    """Build the reviewable, deterministic command used for measured runs."""

    return [
        str(config.executable),
        "--model",
        str(config.model),
        "--prompt",
        config.prompt,
        "--seed",
        str(config.seed),
        "--temp",
        f"{config.temperature:g}",
        "--ctx-size",
        str(config.context_size),
        "--predict",
        str(config.predict_tokens),
        "--gpu-layers",
        config.gpu_layers,
        "--threads",
        str(config.threads),
        "--threads-batch",
        str(config.threads),
        "--conversation",
        "--single-turn",
        "--no-display-prompt",
        "--perf",
        "--log-file",
        str(config.log_file),
    ]
