from pathlib import Path

from expertflow.artifacts import ArtifactSpec
from expertflow.fetching import artifact_destination, fetch_artifact


def test_fetch_uses_repository_scoped_directory(tmp_path: Path) -> None:
    result = artifact_destination(tmp_path, "google/example", "model.gguf")

    assert result == tmp_path / "google--example" / "model.gguf"


def test_fetch_destination_does_not_create_directories(tmp_path: Path) -> None:
    result = artifact_destination(tmp_path, "google/example", "model.gguf")

    assert not result.parent.exists()


def test_fetch_resumes_an_incomplete_canonical_file(tmp_path: Path) -> None:
    spec = ArtifactSpec(
        repository="google/example",
        revision="abc123",
        filename="model.gguf",
        size_bytes=5,
        sha256="9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
    )
    destination = artifact_destination(
        tmp_path, spec.repository, spec.filename
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"mo")
    calls: list[tuple[str, Path]] = []

    def resume(url: str, path: Path) -> None:
        calls.append((url, path))
        path.write_bytes(b"model")

    result = fetch_artifact(spec, tmp_path, downloader=resume)

    assert calls == [
        (
            "https://huggingface.co/google/example/resolve/abc123/model.gguf",
            destination,
        )
    ]
    assert result == destination.resolve()
