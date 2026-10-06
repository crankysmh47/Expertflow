"""Resumable retrieval of pinned external model artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable
from urllib.parse import quote

from expertflow.artifacts import (
    ArtifactSpec,
    ArtifactVerificationError,
    verify_artifact,
)


Downloader = Callable[[str, Path], None]


def artifact_destination(root: Path, repository: str, filename: str) -> Path:
    """Return the deterministic local path for a repository artifact."""

    return root / repository.replace("/", "--") / filename


def artifact_url(spec: ArtifactSpec) -> str:
    """Return the exact revision URL for ``spec`` without branch indirection."""

    repository = quote(spec.repository, safe="/")
    revision = quote(spec.revision, safe="")
    filename = quote(spec.filename, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{filename}"


def download_with_curl(url: str, destination: Path) -> None:
    """Download or resume ``url`` with curl's bounded retry behavior."""

    subprocess.run(
        [
            "curl.exe",
            "--location",
            "--fail",
            "--retry",
            "20",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def fetch_artifact(
    spec: ArtifactSpec,
    root: Path,
    *,
    downloader: Downloader = download_with_curl,
) -> Path:
    """Download or resume ``spec`` into ``root`` and verify it before returning."""

    destination = artifact_destination(root, spec.repository, spec.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            verify_artifact(destination, spec)
            return destination.resolve()
        except ArtifactVerificationError:
            if destination.stat().st_size >= spec.size_bytes:
                raise

    downloader(artifact_url(spec), destination)
    verify_artifact(destination, spec)
    return destination.resolve()
