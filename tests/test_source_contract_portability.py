from pathlib import Path


def test_source_contracts_do_not_embed_private_worktree_paths() -> None:
    for name in ("test_t1_temporal_source_contract.py", "test_t2_sidecar_source_contract.py"):
        text = (Path(__file__).parent / name).read_text(encoding="utf-8")
        assert r"C:\models\expertflow\worktrees" not in text
