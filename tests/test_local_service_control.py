import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_stop_refuses_reused_or_foreign_process_identity(tmp_path):
    from expertflow.product.service_control import process_state, stop_session
    child=subprocess.Popen([sys._base_executable,"-c","import time; time.sleep(60)"],start_new_session=sys.platform != "win32")
    try:
        from expertflow.compiler.runner import _process_creation
        birth,kind=_process_creation(child)
        state={"pid":child.pid,"creation_identity":birth+1,"creation_kind":kind,
               "server":sys._base_executable,"port":8080}
        path=tmp_path / "session-state.json"; path.write_text(json.dumps(state))
        with pytest.raises(ValueError,match="identity|owned|changed"):
            stop_session(tmp_path)
        assert child.poll() is None
    finally:
        child.kill(); child.wait(timeout=5)


def test_verified_stop_ends_owned_process(tmp_path):
    from expertflow.product.service_control import stop_session
    child=subprocess.Popen([sys._base_executable,"-c","import time; time.sleep(60)"],start_new_session=sys.platform != "win32")
    try:
        from expertflow.compiler.runner import _process_creation
        birth,kind=_process_creation(child)
        path=tmp_path / "session-state.json"
        path.write_text(json.dumps({"pid":child.pid,"creation_identity":birth,"creation_kind":kind,"server":sys._base_executable,"port":8080}))
        assert stop_session(tmp_path)["status"] == "STOPPED"
        child.wait(timeout=5)
    finally:
        if child.poll() is None: child.kill(); child.wait(timeout=5)
