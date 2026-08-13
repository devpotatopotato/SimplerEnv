"""Thread-safe JSON/HTTP host for model-specific policy backends."""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .schema import CanonicalAction, ProtocolError, validate_metadata


class PolicyBackend(ABC):
    @property
    @abstractmethod
    def metadata(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def reset(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return None

    @abstractmethod
    def predict(self, request: Mapping[str, Any]) -> list[CanonicalAction]:
        raise NotImplementedError


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend: PolicyBackend):
        super().__init__(address, _PolicyRequestHandler)
        self.backend = backend
        self.inference_lock = threading.Lock()


class _PolicyRequestHandler(BaseHTTPRequestHandler):
    server: PolicyHTTPServer

    def log_message(self, fmt, *args):
        print(f"policy-server: {fmt % args}", flush=True)

    def _write(self, status: int, payload: Mapping[str, Any]):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 256 * 1024 * 1024:
                raise ProtocolError("request body is empty or too large")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("invalid JSON request") from exc
        if not isinstance(value, dict):
            raise ProtocolError("request body must be a JSON object")
        return value

    def do_GET(self):
        try:
            if self.path == "/healthz":
                self._write(200, {"ok": True})
            elif self.path == "/v1/metadata":
                self._write(200, validate_metadata(self.server.backend.metadata))
            else:
                self._write(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001 - exception must cross the process boundary as JSON
            self._write(500, {"error": type(exc).__name__, "detail": str(exc)})

    def do_POST(self):
        try:
            request = self._body()
            if self.path == "/v1/reset":
                with self.server.inference_lock:
                    detail = self.server.backend.reset(request)
                self._write(200, {"ok": True, "detail": detail})
                return
            if self.path != "/v1/actions":
                self._write(404, {"error": "not found"})
                return
            started = time.perf_counter()
            with self.server.inference_lock:
                actions = self.server.backend.predict(request)
            canonical = [item if isinstance(item, CanonicalAction) else CanonicalAction.from_dict(item) for item in actions]
            if not canonical:
                raise ProtocolError("backend returned an empty action chunk")
            self._write(
                200,
                {
                    "actions": [item.to_dict() for item in canonical],
                    "server_inference_ms": (time.perf_counter() - started) * 1000.0,
                },
            )
        except (ProtocolError, ValueError, KeyError, TypeError) as exc:
            self._write(400, {"error": type(exc).__name__, "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - model exceptions must cross the process boundary as JSON
            self._write(500, {"error": type(exc).__name__, "detail": str(exc)})


def serve_backend(backend: PolicyBackend, host: str = "127.0.0.1", port: int = 8000):
    server = PolicyHTTPServer((host, port), backend)
    print(f"Serving {backend.metadata['model_id']} on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
