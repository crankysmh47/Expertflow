from pathlib import Path

import pytest

from expertflow.artifacts import (
    ArtifactSpec,
    ArtifactVerificationError,
    load_artifact_spec,
    verify_artifact,
)


def test_loads_pinned_q4_artifact() -> None:
    spec = load_artifact_spec(Path("configs/model-artifacts.toml"), "gemma4_q4")

    assert spec.repository == "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
    assert spec.revision == "21bfe2a8c89118c9a1a2aa242934fc4d1c0fff15"
    assert spec.filename == "gemma-4-26B_q4_0-it.gguf"
    assert spec.size_bytes == 14_439_361_440
    assert spec.sha256 == "4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5"


def test_rejects_wrong_artifact_size(tmp_path: Path) -> None:
    candidate = tmp_path / "model.gguf"
    candidate.write_bytes(b"wrong")
    spec = ArtifactSpec(
        repository="repo",
        revision="rev",
        filename="model.gguf",
        size_bytes=6,
        sha256="0" * 64,
    )

    with pytest.raises(ArtifactVerificationError, match="size"):
        verify_artifact(candidate, spec)


def test_rejects_wrong_artifact_digest(tmp_path: Path) -> None:
    candidate = tmp_path / "model.gguf"
    candidate.write_bytes(b"model")
    spec = ArtifactSpec(
        repository="repo",
        revision="rev",
        filename="model.gguf",
        size_bytes=5,
        sha256="0" * 64,
    )

    with pytest.raises(ArtifactVerificationError, match="SHA-256"):
        verify_artifact(candidate, spec)
