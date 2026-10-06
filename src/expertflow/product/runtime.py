"""Portable llama.cpp runtime discovery and explicit verified installation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time
from urllib.request import Request, urlopen
from urllib.error import URLError
from urllib.parse import urlparse
import uuid
import zipfile


def file_digest(path: Path, *, deadline=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Absolute identity-verification budget exhausted.")
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment(runtime: dict) -> dict[str, str]:
    # Ambient llama settings must not silently change a stored launch contract.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_ARG_", "GGML_")) and k not in {"LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}}
    directories = [runtime["directory"], *runtime.get("dll_dirs", [])]
    if os.name == "nt":
        system = Path(os.environ.get("SystemRoot", "C:/Windows"))
        env["PATH"] = os.pathsep.join([*directories, str(system / "System32"), str(system)])
    else:
        env["PATH"] = os.pathsep.join([*directories, "/usr/bin", "/bin"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(directories)
    return env


def _binary(directory: Path, name: str) -> Path:
    for candidate in [directory / (name + ".exe"), directory / name]:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("Runtime must contain llama-cli and llama-server. Supply their bin directory.")


def _dependency_files(directory: Path, dll_dirs: list[str], *, deadline=None) -> dict[str, str]:
    files = {}
    for path in directory.iterdir():
        if path.is_file() and (path.suffix.lower() == ".dll" or ".so" in path.name or path.suffix == ".dylib"):
            files[str(path.resolve())] = file_digest(path, deadline=deadline)
    for folder in dll_dirs:
        for path in Path(folder).iterdir():
            if path.is_file() and (path.suffix.lower() == ".dll" or ".so" in path.name or path.suffix == ".dylib"):
                files[str(path.resolve())] = file_digest(path, deadline=deadline)
    return files


def inspect_runtime(directory: Path, *, dll_dirs=(), runner=None) -> dict:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Runtime directory is missing. Supply a directory containing llama-cli and llama-server.")
    runner = subprocess.run if runner is None else runner
    cli, server = _binary(directory, "llama-cli"), _binary(directory, "llama-server")
    dependencies = [str(Path(p).expanduser().resolve()) for p in dll_dirs]
    if any(not Path(p).is_dir() for p in dependencies):
        raise ValueError("DLL dependency directory is missing; correct --dll-dir.")
    result = {"schema_version": 1, "directory": str(directory), "cli": str(cli),
              "server": str(server), "dll_dirs": dependencies, "qualification": "UNTUNED"}
    outputs = {}
    for kind, binary in [("cli", cli), ("server", server)]:
        for flag in ["--version", "--help"]:
            try:
                probe = runner([str(binary), flag], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=20,
                               shell=False, env=runtime_environment(result), cwd=str(directory))
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(f"Cannot probe {binary.name}; check runtime dependencies/DLL directories: {error}") from error
            if probe.returncode:
                raise ValueError(f"{binary.name} probe exited {probe.returncode}. Check runtime dependencies and --dll-dir (CUDA DLLs on Windows).")
            output = (probe.stdout + "\n" + probe.stderr).strip()
            if not output or len(output) > 2 * 1024 * 1024:
                raise ValueError(f"Invalid {binary.name} probe output.")
            outputs[kind + flag] = output
    files = {str(cli): file_digest(cli), str(server): file_digest(server)}
    files.update(_dependency_files(directory, dependencies))
    result.update(version=outputs["server--version"],
                  capabilities=sorted(set(re.findall(r"--[a-z][a-z0-9-]*", outputs["cli--help"] + outputs["server--help"]))),
                  cli_capabilities=sorted(set(re.findall(r"--[a-z][a-z0-9-]*", outputs["cli--help"]))),
                  server_capabilities=sorted(set(re.findall(r"--[a-z][a-z0-9-]*", outputs["server--help"]))),
                  files=files)
    result["identity"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return result


def verify_runtime(runtime: dict, *, deadline=None) -> None:
    try:
        files = runtime["files"]
        if not isinstance(files, dict) or any(runtime[k] not in files for k in ["cli", "server"]):
            raise ValueError("Runtime binary is not registered in its identity manifest; run setup again.")
        if hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() != runtime["identity"]:
            raise ValueError("Runtime identity manifest changed; run setup again.")
        for path, expected in files.items():
            if not Path(path).is_file() or file_digest(Path(path), deadline=deadline) != expected:
                raise ValueError(f"Runtime file changed or missing: {Path(path).name}; register this runtime again.")
        current = _dependency_files(Path(runtime["directory"]), runtime.get("dll_dirs", []), deadline=deadline)
        original = {p: h for p, h in files.items() if p not in {runtime["cli"], runtime["server"]}}
        if current != original:
            raise ValueError("Runtime dependencies changed; register this runtime again.")
    except (KeyError, TypeError, OSError) as error:
        raise ValueError(f"Cannot verify runtime identity: {error}") from error


def download_archive(url: str, destination: Path, sha256: str) -> Path:
    if not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        raise ValueError("A 64-character SHA256 checksum is required.")
    if urlparse(url).scheme not in {"https", "file"}:
        raise ValueError("Downloads require HTTPS (or a local file URL).")
    if destination.exists():
        if file_digest(destination) == sha256.lower():
            return destination
        raise ValueError("Existing archive checksum differs; choose a fresh destination.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urlopen(request, timeout=30) as response:
                resumed = offset and response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an invalid resume range.")
                with partial.open("ab" if resumed else "wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            break
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if file_digest(partial) != sha256.lower():
        raise ValueError("Downloaded archive checksum mismatch; retained .part for diagnosis.")
    partial.replace(destination)
    return destination


def install_archive(archive: Path, destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Install destination exists; choose a fresh directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (destination.name + ".partial-" + uuid.uuid4().hex)
    staging.mkdir()
    total = 0
    def safe_target(name, size):
        nonlocal total
        parts = PurePosixPath(name.replace("\\", "/"))
        target = (staging / str(parts)).resolve()
        if parts.is_absolute() or ".." in parts.parts or ":" in name or not target.is_relative_to(staging):
            raise ValueError("Archive contains an unsafe path.")
        total += size
        if total > 8 * 1024**3:
            raise ValueError("Archive exceeds the 8 GiB extraction limit.")
        return target
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive contains an unsafe symlink.")
                target = safe_target(entry.filename, entry.file_size)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(entry) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    if os.name != "nt":
                        target.chmod((entry.external_attr >> 16) & 0o777 or 0o755)
    else:
        with tarfile.open(archive, "r:*") as bundle:
            for entry in bundle:
                if not entry.isdir() and not entry.isfile():
                    raise ValueError("Archive contains an unsafe link or device.")
                target = safe_target(entry.name, entry.size)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(entry) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(entry.mode & 0o777)
    pairs = [p.parent for p in staging.rglob("llama-server*") if p.name in {"llama-server", "llama-server.exe"}]
    if len(pairs) != 1:
        raise ValueError("Archive must contain exactly one llama-server runtime directory; staging retained.")
    relative = pairs[0].relative_to(staging)
    _binary(pairs[0], "llama-cli")
    staging.rename(destination)
    return destination / relative
