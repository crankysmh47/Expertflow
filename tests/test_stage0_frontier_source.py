from pathlib import Path


def test_stage0_frontier_accepts_explicit_extra_runtime_arguments() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "measure_stage0_frontier.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string[]]$ExtraArgs" in source
    assert "$runtimeArgs += $ExtraArgs" in source
    assert "extra_args = $ExtraArgs" in source
