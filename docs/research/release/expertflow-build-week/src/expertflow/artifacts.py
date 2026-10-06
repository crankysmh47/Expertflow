"""Pinned external artifact metadata and integrity verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tomllib


_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class ArtifactVerificationError(ValueError):
    """Raised when a local artifact does not match its pinned manifest."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Immutable provenance and integrity requirements for one artifact."""

    repository: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str


def load_artifact_spec(path: Path, name: str) -> ArtifactSpec:
    """Load one named artifact from the TOML manifest at ``path``."""

    with path.open("rb") as stream:
        document = tomllib.load(stream)
    record = document["artifacts"][name]
    return ArtifactSpec(
        repository=str(record["repository"]),
        revision=str(record["revision"]),
        filename=str(record["filename"]),
        size_bytes=int(record["size_bytes"]),
        sha256=str(record["sha256"]),
    )


def verify_artifact(path: Path, spec: ArtifactSpec) -> None:
    """Verify the exact byte length and SHA-256 digest of ``path``."""

    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        raise ArtifactVerificationError(
            f"artifact size mismatch: expected {spec.size_bytes}, got {actual_size}"
        )

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    actual_digest = digest.hexdigest()
    if actual_digest != spec.sha256:
        raise ArtifactVerificationError(
            f"artifact SHA-256 mismatch: expected {spec.sha256}, got {actual_digest}"
        )

