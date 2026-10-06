from pathlib import Path
import pytest


def test_alpha_builder_rejects_model_weight_and_native_payload(tmp_path):
    from scripts.build_local_release import audit_payload
    (tmp_path / "weights.gguf").write_bytes(b"private")
    with pytest.raises(ValueError,match="payload"):
        audit_payload(tmp_path)


def test_alpha_checksum_manifest_is_sorted_and_does_not_hash_itself(tmp_path):
    from scripts.build_local_release import checksums
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "SHA256SUMS.json").write_text("ignore")
    result=checksums(tmp_path)
    assert list(result)==["a.txt","b.txt"]
    assert all(len(v)==64 for v in result.values())
