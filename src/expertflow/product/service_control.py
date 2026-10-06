"""Inspect/stop a saved server only after verifying its process birth and image."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import signal


def _read_state(directory: Path) -> dict:
    try:
        state = json.loads((directory / "session-state.json").read_text(encoding="utf-8"))
        if type(state.get("pid")) is not int or state["pid"] <= 0 or type(state.get("creation_identity")) is not int or not isinstance(state.get("server"), str):
            raise ValueError("Invalid server process identity.")
        return state
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise ValueError(f"Cannot read saved session identity: {error}") from error


def _windows_process(state, *, terminate=False) -> bool:
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    api.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    handle = api.OpenProcess(0x1000 | 0x100000 | (0x0001 if terminate else 0), False, state["pid"])
    if not handle:
        if ctypes.get_last_error() == 87:
            return False
        raise ValueError("Cannot inspect owned process identity; stop refused.")
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle, *(ctypes.byref(v) for v in times)):
            raise ValueError("Cannot inspect process birth identity; stop refused.")
        birth = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        image, count = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
        if not api.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(count)):
            raise ValueError("Cannot inspect process image identity; stop refused.")
        if birth != state["creation_identity"] or os.path.normcase(str(Path(image.value).resolve())) != os.path.normcase(str(Path(state["server"]).resolve())):
            raise ValueError("Saved process identity changed or belongs to another process; stop refused.")
        code = wintypes.DWORD()
        if not api.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
            return False
        if terminate:
            if not api.TerminateProcess(handle, 0) or api.WaitForSingleObject(handle, 5000) != 0:
                raise ValueError("Owned process did not stop within five seconds.")
        return True
    finally:
        api.CloseHandle(handle)


def _linux_process(state, *, terminate=False) -> bool:
    if not Path(f"/proc/{state['pid']}").exists():
        return False
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ValueError("Safe process identity control requires Linux pidfd support; use foreground Ctrl+C.")
    try:
        descriptor = os.pidfd_open(state["pid"])
    except ProcessLookupError:
        return False
    try:
        fields = Path(f"/proc/{state['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        image = Path(f"/proc/{state['pid']}/exe").resolve()
        if int(fields[19]) != state["creation_identity"] or image != Path(state["server"]).resolve():
            raise ValueError("Saved process identity changed or is foreign; stop refused.")
        if terminate:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return True
    finally:
        os.close(descriptor)


def process_state(directory: Path) -> dict:
    state = _read_state(directory)
    live = _windows_process(state) if os.name == "nt" else _linux_process(state)
    return {"status": "RUNNING" if live else "STOPPED", "pid": state["pid"], "port": state.get("port"), "session_directory": str(directory)}


def stop_session(directory: Path) -> dict:
    state = _read_state(directory)
    _windows_process(state, terminate=True) if os.name == "nt" else _linux_process(state, terminate=True)
    return {"status": "STOPPED", "pid": state["pid"], "reason": "Owned process identity verified; model/runtime files retained."}
