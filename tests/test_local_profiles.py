import json
import pytest
from local_helpers import write_gguf


def test_profile_roundtrip_and_model_mutation_blocks_launch(tmp_path):
    from expertflow.product.models import inspect_model
    from expertflow.product.profiles import create_profile, save_profile, load_profile, verify_profile
    from expertflow.product.runtime import file_digest
    model_path = write_gguf(tmp_path / "tiny.gguf")
    runtime = {"identity": "a" * 64, "files": {}, "cli": "cli", "server": "server",
               "directory": str(tmp_path), "dll_dirs": [], "capabilities": []}
    profile = create_profile(inspect_model(model_path), runtime, context=4096, gpu_layers=0)
    destination = tmp_path / "profile.json"
    save_profile(destination, profile)
    loaded = load_profile(destination)
    assert loaded["status"] == "RUNNABLE-UNTUNED"
    assert loaded["settings"]["context"] == 4096
    assert loaded["model"]["sha256"] == file_digest(model_path)
    model_path.write_bytes(model_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="model.*changed|Model.*changed"):
        verify_profile(loaded)


def test_profile_never_silently_shrinks_requested_context(tmp_path):
    from expertflow.product.models import inspect_model
    from expertflow.product.profiles import create_profile
    model = inspect_model(write_gguf(tmp_path / "tiny.gguf"))
    with pytest.raises(ValueError, match="context"):
        create_profile(model, {}, context=16384, gpu_layers=0)


def test_profile_revision_preserves_old_configuration(tmp_path):
    from expertflow.product.profiles import save_profile
    target = tmp_path / "profile.json"
    save_profile(target, {"value": 1})
    save_profile(target, {"value": 2})
    assert json.loads(target.read_text())["value"] == 2
    revisions = list(tmp_path.glob("profile.json.*.bak"))
    assert len(revisions) == 1
    assert json.loads(revisions[0].read_text())["value"] == 1


def test_bad_profile_fails_with_input_error_not_traceback(tmp_path):
    from expertflow.product.profiles import load_profile
    target = tmp_path / "profile.json"
    target.write_text('{}')
    with pytest.raises(ValueError, match="profile"):
        load_profile(target)


def test_incomplete_artifact_entry_is_rejected_before_launch():
    from expertflow.product.profiles import verify_profile
    with pytest.raises(ValueError,match="profile|artifact"):
        verify_profile({"model":{"path":"unused","files":[{}]}})


def test_expired_hash_budget_is_timeout_not_invalid_identity(tmp_path):
    from expertflow.product.profiles import create_profile,verify_profile
    from expertflow.product.models import inspect_model
    p=create_profile(inspect_model(write_gguf(tmp_path / "tiny.gguf")),{},context=4096)
    with pytest.raises(TimeoutError,match="budget"):
        verify_profile(p,deadline=0)
