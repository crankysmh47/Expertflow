import json
from pathlib import Path
import socket
import sys
import pytest


def test_server_health_completion_and_owned_cleanup(tmp_path, monkeypatch):
    from expertflow.product.server import ServerSession
    helper = tmp_path / "http helper.py"
    helper.write_text('''import json,sys
from http.server import HTTPServer, BaseHTTPRequestHandler
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  self.send_response(200); self.end_headers()
  self.wfile.write(json.dumps({"status":"ok","default_generation_settings":{"n_ctx":4096}}).encode())
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
  assert data["cache_prompt"] is False and data["temperature"] == 0.0
  self.send_response(200); self.end_headers()
  self.wfile.write(json.dumps({"content":"hello","tokens":[4,5],"tokens_predicted":2,"timings":{"predicted_n":2,"predicted_ms":20,"predicted_per_second":100,"prompt_n":4,"prompt_ms":10}}).encode())
HTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()
''')
    monkeypatch.setattr("expertflow.product.server.verify_profile", lambda profile: None)
    # Windows venv python.exe is a launcher; use the actual interpreter PID so
    # this real HTTP listener exercises direct child socket ownership.
    monkeypatch.setattr("expertflow.product.server.build_command", lambda p, port: [sys._base_executable, str(helper), str(port)])
    p = {"runtime": {"directory": str(tmp_path), "dll_dirs": []}, "settings": {"context":4096}}
    with ServerSession(p, log_dir=tmp_path / "logs", health_timeout=5) as server:
        result = server.complete("test", predict=2)
        assert result["tokens"] == [4,5]
        assert result["timings"]["predicted_per_second"] == 100
        pid = server.child.process.pid
        process = server.child.process
    assert process.poll() is not None
    assert (tmp_path / "logs" / "server.log").exists()


def test_occupied_port_is_not_reused_or_killed(tmp_path):
    from expertflow.product.server import ServerSession
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(ValueError, match="port|Port"):
            with ServerSession({}, port=port, log_dir=tmp_path):
                pass
        assert listener.getsockname()[1] == port


@pytest.mark.parametrize("chat", [False, True])
def test_streaming_deadline_is_absolute_even_with_slow_drip(tmp_path, chat):
    from expertflow.product.server import ServerSession
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    import threading
    import time
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Connection","close")
            self.end_headers()
            try:
                for byte in b'data: {"content":"hello","stop":false}\n\n':
                    self.wfile.write(bytes([byte])); self.wfile.flush(); time.sleep(0.04)
            except (OSError,ConnectionError): pass
    http = ThreadingHTTPServer(("127.0.0.1",0),Handler)
    worker=threading.Thread(target=http.serve_forever,daemon=True); worker.start()
    session=ServerSession({},log_dir=tmp_path)
    session.port=http.server_port
    start= time.monotonic()
    try:
        with pytest.raises((TimeoutError,ValueError,OSError)):
            if chat:
                session.url=f"http://127.0.0.1:{http.server_port}"
                list(session.chat([{"role":"user","content":"test"}],predict=2,timeout=0.25))
            else:
                session.complete("test",predict=2,timeout=0.25,stream_measure=True)
        assert time.monotonic()-start < 0.7
    finally:
        http.shutdown(); http.server_close(); worker.join(2)


def test_verification_cannot_renew_absolute_launch_budget(tmp_path, monkeypatch):
    from expertflow.product.server import ServerSession
    clock=[0.0]
    monkeypatch.setattr("expertflow.product.server.time.monotonic",lambda:clock[0])
    def verify(profile, **kwargs):clock[0]=10.0
    monkeypatch.setattr("expertflow.product.server.verify_profile",verify)
    launches=[]
    monkeypatch.setattr("expertflow.product.server.OwnedProcess.__enter__",lambda self:launches.append(self))
    with pytest.raises(TimeoutError,match="budget|deadline"):
        with ServerSession({},log_dir=tmp_path,deadline=1):pass
    assert launches == []
