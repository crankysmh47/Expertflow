from pathlib import Path

import pytest

from scripts.build_product_release import audit_tree, file_manifest


def test_release_audit_rejects_model_private_path_and_credentials(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("portable", encoding="utf-8")
    assert audit_tree(tmp_path) == []
    (tmp_path / "model.gguf").write_bytes(b"x")
    (tmp_path / "private.txt").write_text(r"C:\models\expertflow\worktrees\private", encoding="utf-8")
    (tmp_path / "secret.env").write_text("OPENAI_API_KEY=sk-test-secret", encoding="utf-8")
    errors = audit_tree(tmp_path)
    assert any("forbidden model" in error for error in errors)
    assert any("private path" in error for error in errors)
    assert any("credential" in error for error in errors)


def test_file_manifest_is_sorted_and_excludes_its_own_hash_file(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "SHA256SUMS.json").write_text("stale", encoding="utf-8")
    rows = file_manifest(tmp_path)
    assert [row["path"] for row in rows] == ["a.txt", "b.txt"]
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_release_builder_is_allowlist_based() -> None:
    source = (Path(__file__).parents[1] / "scripts/build_product_release.py").read_text(encoding="utf-8")
    assert "ALLOWLIST" in source
    assert "format-patch" in source
    assert "ZIP_EPOCH" in source
    assert "shutil.copytree(ROOT" not in source
    assert '"__pycache__"' in source
    assert '"*.pyc"' in source
