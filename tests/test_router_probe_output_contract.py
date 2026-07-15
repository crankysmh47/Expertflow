from pathlib import Path


def test_router_probe_serializes_generated_text_after_generation() -> None:
    source = Path("native/router_probe/main.cpp").read_text(encoding="utf-8")

    assert "generated_text" in source
    assert "llama_token_to_piece" in source
    assert "json_escape" in source
