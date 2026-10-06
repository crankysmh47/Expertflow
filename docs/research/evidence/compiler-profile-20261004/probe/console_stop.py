"""Send CTRL_C only into a private console whose owned PID/creation time match."""
import ctypes
from ctypes import wintypes
import sys
import time

k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
k.CloseHandle.argtypes=[wintypes.HANDLE]
k.AttachConsole.argtypes=[wintypes.DWORD]
k.GenerateConsoleCtrlEvent.argtypes=[wintypes.DWORD,wintypes.DWORD]

def creation(handle):
    fields=[wintypes.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(handle,*(ctypes.byref(f) for f in fields)): raise ctypes.WinError(ctypes.get_last_error())
    return (fields[0].dwHighDateTime<<32)|fields[0].dwLowDateTime

if __name__=='__main__':
    pid,expected=map(int,sys.argv[1:])
    handle=k.OpenProcess(0x1000,False,pid)
    assert handle and creation(handle)==expected,'owned PID/creation mismatch'
    try:
        k.FreeConsole()
        assert k.AttachConsole(pid),'private console attachment failed'
        assert creation(handle)==expected
        assert k.SetConsoleCtrlHandler(None,True)
        assert k.GenerateConsoleCtrlEvent(0,0),'owned console interrupt failed'
        time.sleep(.2)
    finally:
        k.FreeConsole();k.CloseHandle(handle)
