from pathlib import Path

import pytest

from scripts.run_next_layer_predictor import main


def test_test_command_refuses_to_open_test_without_selection_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selection lock"):
        main([
            "test",
            "--manifest", str(tmp_path / "manifest.json"),
            "--output", str(tmp_path),
        ])
