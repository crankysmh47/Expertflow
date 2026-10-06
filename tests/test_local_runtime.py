"""Runtime boundaries: no shell expansion, implicit dependencies or false qualification."""
from pathlib import Path
import subprocess

import pytest


def test_runtime_discovers_pair_and_declared_dlls(tmp_path):
    from expertflow.product.runtime import inspect_runtime
    folder = tmp_path / "runtime space ü"
    folder.mkdir()
    cli = folder / "llama-cli.exe"
    server = folder / "llama-server.exe"
    cli.write_bytes(b"cli")
    server.write_bytes(b"server")
    (folder / "llama.dll").write_bytes(b"lib")
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "version: 10002 (a7312ae9)\n" if argv[-1] == "--version" else "--model --ctx-size --threads --host --port --fit --cpu-moe --no-conversation", "")
    result = inspect_runtime(folder, dll_dirs=[tmp_path], runner=run)
    assert result["cli"] == str(cli.resolve())
    assert result["server"] == str(server.resolve())
    assert result["qualification"] == "UNTUNED"
    assert "--fit" in result["capabilities"]
    assert "llama.dll" in {Path(p).name for p in result["files"]}
    assert all(isinstance(args, list) and kw.get("shell") is False for args, kw in calls)
    assert all(str(tmp_path) in kw["env"]["PATH"] for _, kw in calls)


def test_missing_runtime_pair_has_actionable_error(tmp_path):
    from expertflow.product.runtime import inspect_runtime
    with pytest.raises(ValueError, match="llama-cli.*llama-server"):
        inspect_runtime(tmp_path)


def test_runtime_dependency_change_invalidates_identity(tmp_path):
    from expertflow.product.runtime import inspect_runtime, verify_runtime
    for name in ["llama-cli.exe", "llama-server.exe", "llama.dll"]:
        (tmp_path / name).write_bytes(name.encode())
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "test build --model --ctx-size", "")
    result = inspect_runtime(tmp_path, runner=run)
    (tmp_path / "llama.dll").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_runtime(result)


def test_failed_binary_probe_does_not_inherit_support(tmp_path):
    from expertflow.product.runtime import inspect_runtime
    for name in ["llama-cli.exe", "llama-server.exe"]:
        (tmp_path / name).write_bytes(b"x")
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, -1073741515, "", "")
    with pytest.raises(ValueError, match="DLL|dependencies"):
        inspect_runtime(tmp_path, runner=run)


def test_paths_honor_override_without_creating_on_inspection(tmp_path):
    from expertflow.product.paths import data_root
    target = tmp_path / "custom ü"
    assert data_root({"EXPERTFLOW_HOME": str(target)}) == target
    assert not target.exists()


def test_download_checksum_failure_never_publishes_archive(tmp_path):
    from expertflow.product.runtime import download_archive
    source = tmp_path / "input.zip"
    source.write_bytes(b"wrong")
    target = tmp_path / "output.zip"
    with pytest.raises(ValueError, match="checksum"):
        download_archive(source.as_uri(), target, "0" * 64)
    assert not target.exists()
    assert target.with_suffix(".zip.part").exists()


def test_install_rejects_archive_traversal(tmp_path):
    from expertflow.product.runtime import install_archive
    import zipfile
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "escape")
    with pytest.raises(ValueError, match="unsafe"):
        install_archive(archive, tmp_path / "installed")
    assert not (tmp_path / "escape.txt").exists()


def test_runtime_rejects_unhashed_substituted_server(tmp_path):
    from expertflow.product.runtime import inspect_runtime, verify_runtime
    for name in ["llama-cli.exe", "llama-server.exe"]:
        (tmp_path / name).write_bytes(b"x")
    runtime = inspect_runtime(tmp_path, runner=lambda argv, **kw: subprocess.CompletedProcess(argv,0,"version --model",""))
    other = tmp_path / "other.exe"
    other.write_bytes(b"other")
    runtime["server"] = str(other)
    runtime["files"] = {}
    with pytest.raises(ValueError, match="binary|identity|registered"):
        verify_runtime(runtime)


@pytest.mark.parametrize("mutation", ["change", "add", "remove"])
def test_declared_non_cuda_dependency_mutation_invalidates_runtime(tmp_path, mutation):
    from expertflow.product.runtime import inspect_runtime, verify_runtime
    runtime_dir=tmp_path / "bin";runtime_dir.mkdir()
    dependencies=tmp_path / "deps";dependencies.mkdir()
    library=dependencies / "llama.dll";library.write_bytes(b"original")
    for name in ["llama-cli.exe", "llama-server.exe"]:
        (runtime_dir / name).write_bytes(b"binary")
    runtime=inspect_runtime(runtime_dir,dll_dirs=[dependencies],runner=lambda argv, **kw: subprocess.CompletedProcess(argv,0,"version --model",""))
    if mutation == "change":library.write_bytes(b"changed")
    elif mutation == "remove":library.unlink()
    else:(dependencies / "ggml.dll").write_bytes(b"added")
    with pytest.raises(ValueError,match="changed|missing|dependencies"):
        verify_runtime(runtime)
