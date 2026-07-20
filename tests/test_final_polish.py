import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from expertflow.product import commands
from expertflow.cli.main import main


ROOT = Path(__file__).parents[1]
SCORECARD = ROOT / "release/expertflow-build-week/evidence/release-scorecard.json"


def test_release_scorecard_is_the_documentation_source_of_truth() -> None:
    scorecard = json.loads(SCORECARD.read_text(encoding="utf-8"))
    assert scorecard["metrics"]["single_stream"]["expertflow_decode_tps"] == 28.13
    assert scorecard["metrics"]["single_stream"]["stock_decode_tps"] == 22.967
    assert scorecard["metrics"]["single_stream"]["improvement_pct"] == 22.48
    assert scorecard["metrics"]["placement"]["layers"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20]
    assert scorecard["metrics"]["context"]["allocated_tokens"] == 262144
    assert scorecard["metrics"]["context"]["processed_tokens"] == 417
    assert scorecard["metrics"]["quality"]["strict_ppl_gate_pass"] is False
    assert scorecard["metrics"]["throughput"]["concurrent_outputs_deterministic"] is False
    assert scorecard["metrics"]["single_stream"]["classification"] == "measured"
    assert scorecard["metrics"]["cache_decision"]["classification"] == "simulated"


def test_judge_docs_preserve_headline_metrics_and_caveats() -> None:
    paths = [ROOT / "README.md", ROOT / "JUDGES.md", ROOT / "submission/final-devpost-draft.md"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "28.13" in text, path
        assert "22.967" in text, path
        assert "22.48%" in text, path
        assert "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20" in text, path
        assert "not fully deterministic" in text, path
        assert "417" in text and "262,144" in text, path
        assert "not met" in text, path


def test_portable_judge_scripts_and_replay_ci_exist() -> None:
    required = [
        ".github/workflows/replay.yml",
        "scripts/judge-replay.ps1",
        "scripts/judge-replay.sh",
        "scripts/live-demo.ps1",
        "scripts/verify-release.ps1",
        "scripts/verify-release.sh",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative
    workflow = (ROOT / required[0]).read_text(encoding="utf-8")
    for runner in ("windows-latest", "ubuntu-latest", "macos-latest"):
        assert runner in workflow


def test_dashboard_and_claims_ledger_use_scorecard_precision_and_classes() -> None:
    dashboard = (ROOT / "docs/evidence/product-release/dashboard.html").read_text(encoding="utf-8")
    ledger = (ROOT / "submission/claims-ledger.md").read_text(encoding="utf-8")
    for text in (dashboard, ledger):
        assert "28.13" in text
        assert "22.967" in text
        assert "22.48%" in text
        assert "35.6699" in text
        assert "24.5231" in text
        assert "not fully deterministic" in text
    assert "NO CACHE OPPORTUNITY" in dashboard and "Simulation" in dashboard
    assert "NO CACHE OPPORTUNITY" in ledger and "Simulated" in ledger


def test_readme_relative_links_resolve() -> None:
    for document in (ROOT / "README.md", ROOT / "JUDGES.md"):
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            assert (document.parent / target).resolve().exists(), f"{document}: {target}"


def test_archive_extracts_and_verifies_from_path_with_spaces(tmp_path: Path) -> None:
    destination = tmp_path / "judge path with spaces"
    with zipfile.ZipFile(ROOT / "release/expertflow-build-week.zip") as archive:
        archive.extractall(destination)
    release = destination / "expertflow-build-week"
    verified = subprocess.run(
        [sys.executable, str(release / "scripts/verify_release.py")],
        cwd=release,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert '"status": "pass"' in verified.stdout
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell:
        wrapper = subprocess.run(
            [powershell, "-NoProfile", "-File", str(release / "scripts/verify-release.ps1")],
            cwd=release,
            capture_output=True,
            text=True,
            check=False,
        )
        assert wrapper.returncode == 0, wrapper.stdout + wrapper.stderr
    shell = shutil.which("sh")
    if shell:
        wrapper = subprocess.run(
            [shell, str(release / "scripts/verify-release.sh")],
            cwd=release,
            capture_output=True,
            text=True,
            check=False,
        )
        assert wrapper.returncode == 0, wrapper.stdout + wrapper.stderr


def test_local_visual_and_video_assets_exist() -> None:
    required = [
        "docs/assets/expertflow-logo.png",
        "docs/assets/architecture.svg",
        "docs/assets/placement-map.svg",
        "docs/assets/result.svg",
        "docs/assets/profile-cards.svg",
        "docs/assets/cache-decision.svg",
        "submission/demo-video-script-final.md",
        "submission/demo-video-shot-list-final.md",
        "submission/demo-video-fallback-plan.md",
        "submission/demo-video-assets/title.svg",
        "submission/demo-video-assets/architecture.svg",
        "submission/demo-video-assets/result.svg",
        "submission/demo-video-assets/codex-workflow.svg",
        "submission/demo-video-assets/limitations.svg",
        "submission/demo-video-assets/final-summary.svg",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative


def test_visual_identity_and_codex_attribution_are_explicit() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/assets/expertflow-logo.png" in readme
    assert "GPT-5.6-sol" in readme
    assert "managed the engineering workflow" in readme

    svg_paths = [
        *sorted((ROOT / "docs/assets").glob("*.svg")),
        *sorted((ROOT / "submission/demo-video-assets").glob("*.svg")),
    ]
    for path in svg_paths:
        svg = path.read_text(encoding="utf-8").lower()
        assert "viewbox=" in svg, path
        assert "<title" in svg, path
        assert any(color in svg for color in ("#0b3d20", "#101311", "#d6a84a")), path


def test_doctor_reports_replay_only_with_actionable_cross_platform_status(monkeypatch) -> None:
    monkeypatch.setattr(commands.platform, "system", lambda: "Linux")
    monkeypatch.setattr(commands.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(commands.shutil, "which", lambda name: None)
    report = commands.doctor_report(None, None, None)
    assert report["status"] == "replay_only"
    assert report["exit_code"] == 10
    assert report["live_acceleration_supported"] is False
    assert report["replay_supported"] is True
    assert any(check["name"] == "operating_system_architecture" for check in report["checks"])
    assert all("next_command" in check for check in report["checks"] if check["status"] != "pass")


def test_doctor_resolves_live_paths_from_environment(tmp_path: Path, monkeypatch, capsys) -> None:
    model = tmp_path / "model.gguf"
    cli = tmp_path / "llama-cli.exe"
    server = tmp_path / "llama-server.exe"
    model.write_bytes(b"model")
    cli.write_bytes(b"MZcli")
    server.write_bytes(b"MZserver")
    monkeypatch.setenv("EXPERTFLOW_MODEL_PATH", str(model))
    monkeypatch.setenv("EXPERTFLOW_LLAMA_CLI", str(cli))
    monkeypatch.setenv("EXPERTFLOW_LLAMA_SERVER", str(server))
    assert main(["doctor"]) == 20
    report = json.loads(capsys.readouterr().out)
    by_name = {item["name"]: item for item in report["checks"]}
    assert by_name["model"]["path"] == str(model)
    assert by_name["model"]["bytes"] == 5
    assert by_name["model"]["expected_bytes"] == 22862575520
    assert by_name["model"]["size_match"] is False
    assert by_name["llama_cli"]["path"] == str(cli)
    assert by_name["llama_server"]["path"] == str(server)
