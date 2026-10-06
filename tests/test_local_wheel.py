"""Release artifact gate: explicitly supplied wheel, isolated interpreter/cwd."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest


def test_release_wheel_runs_without_checkout_or_extras(tmp_path):
    artifact = os.environ.get("EXPERTFLOW_TEST_WHEEL")
    if not artifact:
        pytest.skip("Set EXPERTFLOW_TEST_WHEEL for the exact release-artifact gate.")
    wheel = Path(artifact).resolve()
    with zipfile.ZipFile(wheel) as bundle:
        assert "expertflow/product/stock-policy.json" in bundle.namelist()
        assert not any(name.endswith((".gguf", ".exe", ".dll", ".db")) for name in bundle.namelist())
    environment = tmp_path / "wheel env ü"
    subprocess.run([sys.executable,"-m","venv",str(environment)],check=True,capture_output=True)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python),"-m","pip","install","--no-index","--no-deps",str(wheel)],check=True,capture_output=True)
    cwd = tmp_path / "empty cwd"
    cwd.mkdir()
    env = {k:v for k,v in os.environ.items() if k not in {"PYTHONPATH","PYTHONHOME"}}
    env["EXPERTFLOW_HOME"] = str(tmp_path / "untouched state")
    for arguments in [["--help"],["local","--help"],["local","doctor","--json"],["local","run","--help"],["local","verify-job","--help"]]:
        result = subprocess.run([str(python),"-m","expertflow.cli.main",*arguments],cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
        assert result.returncode == 0, result.stderr + result.stdout
        if arguments == ["local","doctor","--json"]:
            assert json.loads(result.stdout)["status"] == "INSPECTED"
    origin = subprocess.run([str(python),"-c","import expertflow.product.local_cli as p; print(p.__file__)"],cwd=cwd,env=env,capture_output=True,text=True,check=True)
    assert str(environment) in origin.stdout
    assert not (tmp_path / "untouched state").exists()
