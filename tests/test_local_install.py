"""Public local CLI works without historical checkout assets."""
import json
from pathlib import Path
import pytest


def test_local_help_has_new_workflow_without_loading_models(capsys):
    from expertflow.cli.main import main
    with pytest.raises(SystemExit) as exit:
        main(["local", "--help"])
    assert exit.value.code == 0
    assert "doctor" in capsys.readouterr().out


def test_local_doctor_missing_runtime_returns_actionable_json(tmp_path, capsys):
    from expertflow.cli.main import main
    result = main(["local", "doctor", "--runtime", str(tmp_path / "missing"), "--json"])
    assert result == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "UNSUPPORTED"
    assert "runtime" in output["reason"].lower()
    assert output["next_action"]


def test_local_doctor_without_model_is_read_only(tmp_path, capsys, monkeypatch):
    from expertflow.cli.main import main
    monkeypatch.setenv("EXPERTFLOW_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    assert main(["local", "doctor", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["hardware"]["system_ram_bytes"] > 0
    assert not (tmp_path / "state").exists()
