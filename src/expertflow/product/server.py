"""Loopback llama-server sessions, health, completion and streaming chat."""
from __future__ import annotations
import json
import http.client
from pathlib import Path
import socket
import time
import threading
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import URLError

from .profiles import verify_profile
from .runtime import runtime_environment
from .session import OwnedProcess, build_command


class ServerSession:
    def __init__(self, profile: dict, *, log_dir: Path, port=0, health_timeout=180, deadline=None):
        self.profile, self.log_dir, self.port = profile, log_dir, port
        self.health_timeout = health_timeout
        self.deadline = deadline
        self.child = None
        self.props = None
        self.opener = build_opener(ProxyHandler({}))

    def __enter__(self):
        def check_budget():
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise TimeoutError("Absolute job budget exhausted before native launch.")
        check_budget()
        if type(self.port) is not int or not 0 <= self.port <= 65535:
            raise ValueError("Port must be 0 (automatic) or 1–65535.")
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", self.port))
                self.port = probe.getsockname()[1]
            except OSError as error:
                raise ValueError(f"Port {self.port} is occupied; choose another --port. No existing process was stopped.") from error
        if self.deadline is None:
            verify_profile(self.profile)
        else:
            verify_profile(self.profile, deadline=self.deadline)
        check_budget()
        self.url = f"http://127.0.0.1:{self.port}"
        self.child = OwnedProcess(build_command(self.profile, port=self.port),
                                  env=runtime_environment(self.profile["runtime"]),
                                  log_path=self.log_dir / "server.log", deadline=self.deadline)
        self.child.__enter__()
        deadline = time.monotonic() + self.health_timeout
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        try:
            while time.monotonic() < deadline:
                if self.child.process.poll() is not None:
                    raise ValueError(f"Model load failed (exit {self.child.process.returncode}); inspect {self.log_dir / 'server.log'}. Check architecture, RAM/VRAM, context and runtime dependencies.")
                try:
                    if self.request("/health", timeout=min(2, max(0.001, deadline-time.monotonic()))).get("status") == "ok":
                        from expertflow.compiler.runner import _owns_port
                        if not _owns_port(self.child.process.pid, self.port):
                            raise ValueError("Health endpoint is not owned by the launched process; server rejected.")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Absolute startup deadline exceeded.")
                        self.props = self.request("/props", timeout=min(5, deadline-time.monotonic()))
                        actual = self.props.get("default_generation_settings", {}).get("n_ctx")
                        if actual is not None and actual < self.profile["settings"]["context"]:
                            raise ValueError("Runtime reduced the requested context; choose an explicit smaller profile.")
                        launch = json.loads((self.log_dir / "launch.json").read_text(encoding="utf-8"))
                        (self.log_dir / "session-state.json").write_text(json.dumps({
                            "pid": self.child.process.pid, "creation_identity": launch["creation_identity"],
                            "creation_kind": launch["creation_kind"],
                            "server": self.child.command[0], "port": self.port}, indent=2)+"\n", encoding="utf-8")
                        return self
                except (URLError, TimeoutError, OSError):
                    pass
                time.sleep(0.1)
            raise ValueError(f"Model health timed out; inspect {self.log_dir / 'server.log'}. Reduce context explicitly or check runtime dependencies.")
        except BaseException:
            self.child.__exit__()
            raise

    def __exit__(self, *exc):
        if self.child:
            self.child.__exit__(*exc)

    def request(self, route: str, payload=None, *, timeout=120) -> dict:
        from expertflow.compiler.runner import _http
        result = _http(self.port, route, payload, timeout=timeout)
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError(f"Runtime request failed: {result.get('error') if isinstance(result, dict) else 'invalid response'}")
        return result

    def complete(self, prompt: str, *, predict=128, timeout=120, ignore_eos=False, stream_measure=False) -> dict:
        if not isinstance(prompt, str) or not prompt or len(prompt.encode()) > 1024 * 1024:
            raise ValueError("Prompt must contain 1 byte to 1 MiB of text.")
        if type(predict) is not int or not 1 <= predict <= 65536:
            raise ValueError("Predict tokens must be between 1 and 65536.")
        start = time.monotonic()
        payload = {"prompt": prompt, "n_predict": predict, "seed": 42,
            "temperature": 0.0, "cache_prompt": False, "return_tokens": True,
            "ignore_eos": ignore_eos, "stream": stream_measure}
        result = self._stream_completion(payload, timeout) if stream_measure else self.request("/completion", payload, timeout=timeout)
        result["wall_seconds"] = time.monotonic() - start
        return result

    def _stream_completion(self, payload: dict, timeout: float) -> dict:
        start = time.monotonic()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        expired = threading.Event()
        owned_socket = []
        def expire():
            expired.set()
            if owned_socket:
                try:
                    owned_socket[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        first, content, tokens, final = None, [], [], None
        try:
            connection.connect()
            owned_socket.append(connection.sock)
            if expired.is_set():
                raise TimeoutError("Absolute streaming completion deadline exceeded.")
            connection.request("POST", "/completion", json.dumps(payload).encode(), {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f"Completion failed: HTTP {response.status}.")
            total = 0
            while not expired.is_set():
                line = response.readline(1024 * 1024 + 1)
                total += len(line)
                if len(line) > 1024 * 1024 or total > 32 * 1024**2:
                    raise ValueError("Streaming measurement exceeded response bounds.")
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("error"):
                    raise ValueError(f"Completion failed: {event['error']}")
                if first is None and (event.get("tokens") or event.get("content")):
                    first = time.monotonic() - start
                if event.get("stop"):
                    final = event
                    break
                content.append(event.get("content", ""))
                tokens.extend(event.get("tokens", []))
            if expired.is_set():
                raise TimeoutError("Absolute streaming completion deadline exceeded.")
            if final is None:
                raise ValueError("Completion stream ended without a final receipt.")
            final["ttft_seconds"] = first
            if not final.get("tokens"):
                final["tokens"] = tokens
            final["content"] = "".join(content)
            return final
        finally:
            timer.cancel()
            connection.close()

    def chat(self, messages: list[dict], *, predict=256, timeout=120):
        if type(predict) is not int or not 1 <= predict <= 65536:
            raise ValueError("Predict tokens must be between 1 and 65536.")
        if not isinstance(messages, list) or not messages or len(json.dumps(messages).encode()) > 1024**2:
            raise ValueError("Chat history must contain messages and fit within 1 MiB.")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        expired, owned_socket = threading.Event(), []
        def expire():
            expired.set()
            if owned_socket:
                try:
                    owned_socket[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        payload = json.dumps({
            "messages": messages, "max_tokens": predict, "seed": 42,
            "temperature": 0.0, "stream": True}).encode()
        try:
            connection.connect()
            owned_socket.append(connection.sock)
            if expired.is_set():
                raise TimeoutError("Absolute chat deadline exceeded.")
            connection.request("POST", "/v1/chat/completions", payload, {"Content-Type":"application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f"Chat failed: HTTP {response.status}.")
            total, done, has_text, has_reasoning, finish = 0, False, False, False, None
            while not expired.is_set():
                line = response.readline(1024 * 1024 + 1)
                total += len(line)
                if not line:
                    break
                if len(line) > 1024 * 1024 or total > 32 * 1024**2:
                    raise ValueError("Oversized streaming response.")
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError(f"Runtime stream failed: {event['error']}")
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    has_reasoning = has_reasoning or bool(delta.get("reasoning_content"))
                    finish = choice.get("finish_reason") or finish
                    text = delta.get("content")
                    if text:
                        has_text = True
                        yield text
            if expired.is_set():
                raise TimeoutError("Absolute chat deadline exceeded.")
            if not done:
                raise ValueError("Chat stream ended without a final receipt.")
            if not has_text:
                detail = "Reasoning consumed the answer budget" if has_reasoning else "No answer text was returned"
                raise ValueError(f"{detail} (finish: {finish}); try a larger explicit --predict budget or inspect the model chat template.")
        finally:
            timer.cancel()
            connection.close()
