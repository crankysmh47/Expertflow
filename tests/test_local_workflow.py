import json
import pytest
from local_helpers import write_gguf


def test_profiles_remove_deletes_only_a_valid_profile(tmp_path, capsys):
    from expertflow.cli.main import main
    from expertflow.product.models import inspect_model
    from expertflow.product.profiles import create_profile, save_profile
    model=write_gguf(tmp_path / "tiny.gguf")
    profile=tmp_path / "p.json"
    save_profile(profile,create_profile(inspect_model(model),{},context=4096))
    assert main(["local","profiles","remove","--profile",str(profile),"--json"]) == 0
    assert not profile.exists()
    assert model.exists()


def test_run_help_exposes_prompt_or_interactive_workflow(capsys):
    from expertflow.cli.main import main
    with pytest.raises(SystemExit) as result:
        main(["local","run","--help"])
    assert result.value.code == 0
    assert "--prompt" in capsys.readouterr().out


def test_tune_unsupported_emits_result_and_keeps_profile(tmp_path, monkeypatch, capsys):
    from expertflow.cli.main import main
    from expertflow.product.models import inspect_model
    from expertflow.product.profiles import create_profile, save_profile
    model=write_gguf(tmp_path / "tiny.gguf")
    profile=tmp_path / "p.json"
    save_profile(profile,create_profile(inspect_model(model),{},context=4096))
    before=profile.read_bytes()
    code=main(["local","tune","--profile",str(profile),"--output-dir",str(tmp_path / "job"),"--json"])
    report=json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["status"] == "UNSUPPORTED"
    assert report["model_processes"] == 0
    assert profile.read_bytes() == before


def test_whitespace_one_shot_prompt_exits_without_native_launch(tmp_path,monkeypatch,capsys):
    from expertflow.cli.main import main
    from expertflow.product.models import inspect_model
    from expertflow.product.profiles import create_profile,save_profile
    path=tmp_path / "p.json"
    save_profile(path,create_profile(inspect_model(write_gguf(tmp_path / "tiny.gguf")),{},context=4096))
    monkeypatch.setattr("expertflow.product.server.ServerSession.__enter__",lambda self:pytest.fail("No model should launch"))
    assert main(["local","run","--profile",str(path),"--prompt","   ","--json"]) == 2
