"""Validated argv construction and lifetime of a single owned runtime process."""
from __future__ import annotations
import os
import json
from pathlib import Path
import signal
import subprocess
import time


def build_command(profile: dict, *, port: int) -> list[str]:
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    settings = profile["settings"]
    command = [profile["runtime"]["server"], "--model", profile["model"]["path"],
               "--ctx-size", str(settings["context"]), "--host", "127.0.0.1",
               "--port", str(port), "--parallel", "1"]
    capabilities = profile["runtime"].get("server_capabilities", [])
    for flag, value in [("--n-gpu-layers", settings.get("gpu_layers")), ("--threads", settings.get("threads"))]:
        if value is not None:
            command.extend([flag, str(value)])
    if settings.get("cpu_moe"):
        if "--cpu-moe" not in capabilities:
            raise ValueError("This runtime does not support --cpu-moe; choose an explicit supported profile.")
        command.append("--cpu-moe")
    for key, flag in [("kv_type_k", "--cache-type-k"), ("kv_type_v", "--cache-type-v")]:
        if flag in capabilities:
            command.extend([flag, settings[key]])
    if "--fit" in capabilities:
        command.extend(["--fit", "on"])
    return command


class OwnedProcess:
    def __init__(self, command: list[str], *, env: dict | None, log_path: Path, deadline=None):
        self.command, self.env, self.log_path = command, env, log_path
        self.process = None
        self.log = None
        self.started = None
        self.deadline = deadline
        self.job = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("xb")
        self.started = time.monotonic()
        try:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise TimeoutError("Absolute job budget exhausted before process creation.")
            if os.name == "nt":
                from .windows_job import WindowsJob
                self.job = WindowsJob()
            self.process = subprocess.Popen(self.command, env=self.env, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT, shell=False,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x4 if os.name == "nt" else 0,
                start_new_session=os.name != "nt")
            if self.job:
                self.job.attach_and_resume(self.process, deadline=self.deadline)
            from expertflow.compiler.runner import _process_creation
            birth, kind = _process_creation(self.process)
            (self.log_path.parent / "launch.json").write_text(json.dumps({
                "pid":self.process.pid, "creation_identity":birth, "creation_kind":kind,
                "command":self.command, "started_at":time.time()},indent=2)+"\n",encoding="utf-8")
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        try:
            if self.job:
                self.job.close()
            if self.process is not None and self.process.poll() is None:
                if os.name == "nt":
                    self.process.kill()  # retained OS process handle, never a reused PID
                else:
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self.log is not None:
                self.log.close()
            if self.process is not None:
                (self.log_path.parent / "process-receipt.json").write_text(json.dumps({
                    "pid":self.process.pid,"exit_code":self.process.poll(),
                    "cleanup":self.process.poll() is not None,
                    "wall_seconds":time.monotonic()-self.started},indent=2)+"\n",encoding="utf-8")
