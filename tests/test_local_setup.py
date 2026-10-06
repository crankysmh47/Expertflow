import json
from contextlib import contextmanager
import pytest
from local_helpers import write_gguf


def test_setup_persists_failed_probe_without_claiming_runnable(tmp_path, monkeypatch):
    from expertflow.product.setup import setup_profile
    model = write_gguf(tmp_path / "tiny.gguf")
    runtime = {"version": "test", "directory": str(tmp_path)}
    monkeypatch.setattr("expertflow.product.setup.inspect_runtime", lambda *a, **kw: runtime)
    @contextmanager
    def failing(*a, **kw):
        raise ValueError("OOM")
        yield
    monkeypatch.setattr("expertflow.product.setup.ServerSession", failing)
    output = tmp_path / "profile.json"
    with pytest.raises(ValueError, match="OOM"):
        setup_profile(model, tmp_path, output, context=4096)
    saved = json.loads(output.read_text())
    assert saved["load_verified"] is False
    assert saved["status"] == "ENVIRONMENT-BLOCKED"
    assert saved["load_error"] == "OOM"


def test_setup_success_retains_load_receipt_and_requested_settings(tmp_path, monkeypatch):
    from expertflow.product.setup import setup_profile
    model = write_gguf(tmp_path / "tiny.gguf")
    monkeypatch.setattr("expertflow.product.setup.inspect_runtime", lambda *a, **kw: {"version":"test"})
    class Server:
        props = {"default_generation_settings":{"n_ctx":4096}}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def complete(self,*args,**kw):
            return {"content":"hello","tokens_predicted":1,"tokens":[42],"timings":{"predicted_n":1},"wall_seconds":0.1}
    monkeypatch.setattr("expertflow.product.setup.ServerSession", lambda *a,**kw: Server())
    output = tmp_path / "profile.json"
    result = setup_profile(model, tmp_path, output, context=4096)
    assert result["load_verified"] is True
    assert result["status"] == "RUNNABLE-UNTUNED"
    assert result["settings"]["context"] == 4096
    assert result["load_receipt"]["tokens_predicted"] == 1


def test_setup_without_probe_is_explicitly_unverified(tmp_path, monkeypatch):
    from expertflow.product.setup import setup_profile
    monkeypatch.setattr("expertflow.product.setup.inspect_runtime", lambda *a,**kw: {})
    result = setup_profile(write_gguf(tmp_path / "tiny.gguf"), tmp_path, tmp_path / "p.json", context=4096, probe=False)
    assert result["load_verified"] is False
    assert result["status"] == "UNVERIFIED-PROFILE"
