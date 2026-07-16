from pathlib import Path

import pytest

from scripts.export_temporal_runtime_predictor import export_temporal_bundle


def test_export_refuses_non_temporal_or_missing_lock(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        export_temporal_bundle(
            lock_path=tmp_path / "selection-lock.json",
            manifest_path=tmp_path / "manifest.json",
            output=tmp_path / "output",
        )

