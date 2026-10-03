"""Throwaway proof that a hidden private console permits graceful child shutdown."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).parent))
from console_stop import creation

info=subprocess.STARTUPINFO();info.dwFlags=subprocess.STARTF_USESHOWWINDOW;info.wShowWindow=0
p=subprocess.Popen([sys.executable,'-c',"import signal,time,sys; signal.signal(signal.SIGINT,lambda s,f:sys.exit(0)); print('ready',flush=True); time.sleep(60)"],
    creationflags=subprocess.CREATE_NEW_CONSOLE,startupinfo=info,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:
    assert p.stdout.readline().strip()==b'ready'
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('console_stop.py')),str(p.pid),str(creation(wintypes.HANDLE(int(p._handle))))],
        creationflags=subprocess.CREATE_NO_WINDOW,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert p.wait(timeout=10)==0
    print('Hidden private-console graceful shutdown passed; only owned child targeted')
finally:
    if p.poll() is None: p.kill();p.wait()
