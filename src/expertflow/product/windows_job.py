"""Own a Windows process tree before its primary thread can execute."""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import time


class WindowsJob:
    def __init__(self):
        api=self.api=ctypes.WinDLL("kernel32",use_last_error=True)
        api.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
        api.CreateJobObjectW.restype=wintypes.HANDLE
        api.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        api.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        api.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
        api.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
        api.Thread32First.argtypes=[wintypes.HANDLE,ctypes.c_void_p]
        api.Thread32Next.argtypes=[wintypes.HANDLE,ctypes.c_void_p]
        api.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        api.OpenThread.restype=wintypes.HANDLE
        api.ResumeThread.argtypes=[wintypes.HANDLE]
        api.ResumeThread.restype=wintypes.DWORD
        class Basic(ctypes.Structure):
            _fields_=[("process_time",ctypes.c_longlong),("job_time",ctypes.c_longlong),
                      ("flags",wintypes.DWORD),("minimum",ctypes.c_size_t),("maximum",ctypes.c_size_t),
                      ("active",wintypes.DWORD),("affinity",ctypes.c_size_t),
                      ("priority",wintypes.DWORD),("scheduling",wintypes.DWORD)]
        class Limits(ctypes.Structure):
            _fields_=[("basic",Basic),("io",ctypes.c_ulonglong*6),
                      ("process_memory",ctypes.c_size_t),("job_memory",ctypes.c_size_t),
                      ("peak_process",ctypes.c_size_t),("peak_job",ctypes.c_size_t)]
        self.handle=api.CreateJobObjectW(None,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits=Limits();limits.basic.flags=0x2000  # KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error=ctypes.get_last_error();self.close();raise ctypes.WinError(error)

    def attach_and_resume(self,process, *, deadline=None):
        api=self.api
        if not api.AssignProcessToJobObject(self.handle,int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        class ThreadEntry(ctypes.Structure):
            _fields_=[("size",wintypes.DWORD),("usage",wintypes.DWORD),
                      ("tid",wintypes.DWORD),("pid",wintypes.DWORD),
                      ("base_priority",wintypes.LONG),("delta_priority",wintypes.LONG),("flags",wintypes.DWORD)]
        snapshot=api.CreateToolhelp32Snapshot(0x4,0)  # SNAPTHREAD
        if snapshot in {None,ctypes.c_void_p(-1).value}:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry=ThreadEntry();entry.size=ctypes.sizeof(entry)
            found=api.Thread32First(snapshot,ctypes.byref(entry))
            while found:
                if entry.pid == process.pid:
                    thread=api.OpenThread(0x2,False,entry.tid)  # SUSPEND_RESUME
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if deadline is not None and time.monotonic() >= deadline:
                            raise TimeoutError("Absolute launch budget exhausted before owned thread resume.")
                        if api.ResumeThread(thread) == 0xffffffff:
                            raise ctypes.WinError(ctypes.get_last_error())
                        return
                    finally:
                        api.CloseHandle(thread)
                entry.size=ctypes.sizeof(entry)
                found=api.Thread32Next(snapshot,ctypes.byref(entry))
            raise OSError("Suspended primary thread not found; owned process will be terminated.")
        finally:
            api.CloseHandle(snapshot)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle=None
