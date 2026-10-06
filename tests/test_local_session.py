import subprocess
import sys
import pytest
import os


def profile(tmp_path):
    return {"runtime": {"server": str(tmp_path / 'runtime & echo injected.exe'),
                        "cli": "cli", "server_capabilities": ["--fit", "--cpu-moe", "--cache-type-k", "--cache-type-v"]},
            "model": {"path": str(tmp_path / 'model & echo injected.gguf')},
            "settings": {"context": 4096, "gpu_layers": None, "threads": None, "cpu_moe": False,
                         "kv_type_k": "f16", "kv_type_v": "f16", "seed": 42, "temperature": 0.0}}


def test_command_preserves_paths_and_context_without_shell_expansion(tmp_path):
    from expertflow.product.session import build_command
    p = profile(tmp_path)
    command = build_command(p, port=8080)
    assert command[0] == p["runtime"]["server"]
    assert command[command.index("--model") + 1] == p["model"]["path"]
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--threads" not in command
    assert "--n-gpu-layers" not in command


def test_unsupported_explicit_cpu_moe_fails_instead_of_silent_fallback(tmp_path):
    from expertflow.product.session import build_command
    p = profile(tmp_path)
    p["settings"]["cpu_moe"] = True
    p["runtime"]["server_capabilities"] = []
    with pytest.raises(ValueError, match="cpu-moe"):
        build_command(p, port=8080)


def test_owned_process_timeout_stops_only_its_child(tmp_path):
    from expertflow.product.session import OwnedProcess
    child = OwnedProcess([sys.executable, "-c", "import time; time.sleep(60)"], env=None, log_path=tmp_path / "child.log")
    with child:
        assert child.process.poll() is None
    assert child.process.poll() is not None
    assert (tmp_path / "child.log").exists()
    import json
    launch=json.loads((tmp_path / "launch.json").read_text())
    receipt=json.loads((tmp_path / "process-receipt.json").read_text())
    assert launch["pid"] == child.process.pid
    assert launch["creation_identity"]
    assert receipt["cleanup"] is True


@pytest.mark.skipif(os.name != "nt",reason="Windows process handle gate")
def test_windows_cleanup_uses_owned_handles_not_pid_tree_commands(tmp_path,monkeypatch):
    from expertflow.product.session import OwnedProcess
    def refuse(*args, **kwargs):
        pytest.fail("PID-based taskkill can race process reuse; use the owned process/job handles")
    monkeypatch.setattr("expertflow.product.session.subprocess.run",refuse)
    with OwnedProcess([sys._base_executable,"-c","import time; time.sleep(60)"],env=None,log_path=tmp_path / "child.log") as child:
        assert child.process.poll() is None
    assert child.process.poll() is not None


@pytest.mark.skipif(os.name != "nt",reason="Windows descendant job gate")
def test_windows_job_stops_immediate_descendant_and_keeps_foreign_process(tmp_path):
    import ctypes
    from ctypes import wintypes
    import time
    from expertflow.product.session import OwnedProcess
    control=subprocess.Popen([sys._base_executable,"-c","import time;time.sleep(60)"])
    api=ctypes.WinDLL("kernel32",use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=None
    try:
        code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
        pid_file=tmp_path / "descendant.pid"
        with OwnedProcess([sys._base_executable,"-c",code,str(pid_file)],env=None,log_path=tmp_path / "tree.log"):
            deadline=time.monotonic()+5
            while not pid_file.exists() and time.monotonic()<deadline:time.sleep(0.01)
            assert pid_file.exists()
            handle=api.OpenProcess(0x100000,False,int(pid_file.read_text()))
            assert handle
        assert api.WaitForSingleObject(handle,5000) == 0
        assert control.poll() is None
    finally:
        if handle:api.CloseHandle(handle)
        control.kill();control.wait(timeout=5)
