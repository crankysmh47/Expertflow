from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading
import pytest


def test_reasoning_only_stream_reports_exhausted_answer_budget(tmp_path):
    from expertflow.product.server import ServerSession
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200);self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}\n\n'
                              b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                              b'data: [DONE]\n\n')
    http=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
    session=ServerSession({},log_dir=tmp_path);session.port=http.server_port
    try:
        with pytest.raises(ValueError,match="reasoning|predict|budget"):
            list(session.chat([{"role":"user","content":"hi"}],predict=2,timeout=2))
    finally:
        http.shutdown();http.server_close();worker.join(2)
