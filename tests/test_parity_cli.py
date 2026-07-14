import json
from pathlib import Path

from expertflow.cli.main import main


def write_sequence(path: Path, generated: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "prompt_token_ids": [1, 2],
                "generated_token_ids": generated,
            }
        ),
        encoding="utf-8",
    )


def test_parity_cli_writes_measured_report(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    instrumented = tmp_path / "instrumented.json"
    output = tmp_path / "parity.json"
    write_sequence(baseline, [3, 4])
    write_sequence(instrumented, [3, 4])

    result = main(
        [
            "parity",
            str(baseline),
            str(instrumented),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert report["classification"] == "measured"
    assert report["generated_matches"] is True


def test_parity_cli_fails_the_gate_on_token_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    instrumented = tmp_path / "instrumented.json"
    output = tmp_path / "parity.json"
    write_sequence(baseline, [3, 4])
    write_sequence(instrumented, [3, 9])

    result = main(
        [
            "parity",
            str(baseline),
            str(instrumented),
            "--output",
            str(output),
        ]
    )

    assert result == 1
    assert json.loads(output.read_text(encoding="utf-8"))[
        "generated_matches"
    ] is False
