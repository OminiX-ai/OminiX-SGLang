#!/usr/bin/env python3
"""Cross-repo contract check: standard OminiX-API -> worker-v0 HTTP/SSE.

Set OMINIX_API_BIN to a built OminiX-API executable. The test worker exposes
the same authenticated production control and streaming contract as the gRPC
shim while keeping this check independent of CUDA/model availability.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TOKEN = "bridge-check-token"
PUBLIC_MODEL = "public-c2rust"
SERVED_MODEL = "C2Rust-FP8-DFlash"
PROTOCOL_VERSION = "ominix.worker.v0"
SHIM_BACKEND = "sglang-ominix-v0-http-shim"


class ContractWorker(ThreadingHTTPServer):
    last_generate: dict[str, Any] | None = None


class ContractWorkerHandler(BaseHTTPRequestHandler):
    server: ContractWorker

    def do_GET(self) -> None:
        if not self._authorize():
            return
        if self.path == "/server_info":
            self._send_json(
                HTTPStatus.OK,
                {
                    "scheduler_backend": SHIM_BACKEND,
                    "mode": "grpc",
                    "protocol_version": PROTOCOL_VERSION,
                    "routes": {"generate": "/generate"},
                    "public_openai_api": False,
                    "auth_required": True,
                },
            )
            return
        if self.path == "/get_model_info":
            self._send_json(
                HTTPStatus.OK,
                {"served_model_name": SERVED_MODEL, "is_generation": True},
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if not self._authorize():
            return
        if self.path == "/abort_request":
            request = self._read_json()
            self._send_json(
                HTTPStatus.OK,
                {"success": True, "aborted": request.get("request_id")},
            )
            return
        if self.path != "/generate":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return

        request = self._read_json()
        if request.get("model") != SERVED_MODEL:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "wrong model"}},
            )
            return
        self.server.last_generate = request
        request_id = str(request["request_id"])
        events = [
            _worker_event(request_id, "token", text_delta="po"),
            _worker_event(
                request_id,
                "usage",
                usage={
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            ),
            _worker_event(
                request_id,
                "done",
                text_delta="ng",
                finish_reason="stop",
            ),
        ]
        body = "".join(
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n" for event in events
        )
        body += "data: [DONE]\n\n"
        raw = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _authorize(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {TOKEN}":
            return True
        self._send_json(
            HTTPStatus.UNAUTHORIZED,
            {"error": {"message": "unauthorized"}},
        )
        return False

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("expected JSON object")
        return value

    def _send_json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class OminiXApiToV0ShimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sglang_repo = Path(__file__).resolve().parents[2]
        api_binary = os.environ.get("OMINIX_API_BIN")
        if api_binary:
            self.api_binary = Path(api_binary).expanduser().resolve()
        else:
            self.api_binary = (
                self.sglang_repo.parent / "OminiX-API/target/debug/ominix-api"
            )
        if not self.api_binary.is_file():
            self.skipTest(
                "set OMINIX_API_BIN to a built standard OminiX-API executable"
            )

        self.worker = ContractWorker(("127.0.0.1", 0), ContractWorkerHandler)
        self.worker_thread = threading.Thread(
            target=self.worker.serve_forever,
            daemon=True,
        )
        self.worker_thread.start()
        self.addCleanup(self._stop_worker)

        self.api_port = _free_port()
        self.temp_home = tempfile.TemporaryDirectory(prefix="ominix-api-contract-")
        self.addCleanup(self.temp_home.cleanup)
        self.api_proc = self._start_api()
        self.addCleanup(_stop_process, self.api_proc)
        self.base_url = f"http://127.0.0.1:{self.api_port}"
        _wait_for_status(f"{self.base_url}/readyz", self.api_proc, HTTPStatus.OK)

    def test_nonstream_stream_and_exact_worker_identity(self):
        models = _get_json(f"{self.base_url}/v1/models")
        self.assertIn(PUBLIC_MODEL, [item["id"] for item in models["data"]])

        chat = _post_json(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": PUBLIC_MODEL,
                "messages": [{"role": "user", "content": "say pong"}],
                "max_tokens": 4,
                "stream": False,
            },
        )
        self.assertEqual(chat["choices"][0]["message"]["content"], "pong")
        self.assertEqual(chat["model"], PUBLIC_MODEL)
        self.assertEqual(self.worker.last_generate["model"], SERVED_MODEL)
        self.assertFalse(self.worker.last_generate["stream"])

        body = _post_raw(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": PUBLIC_MODEL,
                "messages": [{"role": "user", "content": "stream pong"}],
                "max_tokens": 4,
                "stream": True,
            },
            accept="text/event-stream",
        )
        self.assertEqual(_stream_text(body), "pong")
        self.assertIn('"finish_reason":"stop"', body)
        self.assertIn("data: [DONE]", body)

    def test_mapped_model_fails_closed_when_worker_stops(self):
        self._stop_worker()
        with self.assertRaises(urllib.error.HTTPError) as raised:
            _post_json(
                f"{self.base_url}/v1/chat/completions",
                {
                    "model": PUBLIC_MODEL,
                    "messages": [{"role": "user", "content": "say pong"}],
                    "max_tokens": 4,
                },
            )
        self.assertEqual(raised.exception.code, HTTPStatus.SERVICE_UNAVAILABLE)

    def _start_api(self) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env.update(
            {
                "HOME": self.temp_home.name,
                "PORT": str(self.api_port),
                "OMINIX_API_HOST": "127.0.0.1",
                "LLM_MODEL": "",
                "RUST_LOG": "warn",
                "OMINIX_V0_SCHEDULER_URL": (
                    f"http://127.0.0.1:{self.worker.server_port}"
                ),
                "OMINIX_V0_SCHEDULER_MODELS": PUBLIC_MODEL,
                "OMINIX_V0_SERVED_MODEL": SERVED_MODEL,
                "OMINIX_V0_SCHEDULER_TOKEN": TOKEN,
                "OMINIX_V0_SCHEDULER_TIMEOUT_SECS": "5",
            }
        )
        return subprocess.Popen(
            [str(self.api_binary)],
            cwd=self.api_binary.parent,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _stop_worker(self) -> None:
        if getattr(self, "worker", None) is None:
            return
        self.worker.shutdown()
        self.worker.server_close()
        self.worker_thread.join(timeout=5)
        self.worker = None


def _worker_event(request_id: str, kind: str, **payload: Any) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "WorkerEvent",
        "request_id": request_id,
        "kind": kind,
        "created_at_ms": 1,
        "sequence_index": 0,
        **payload,
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _wait_for_status(
    url: str,
    proc: subprocess.Popen[str],
    expected: HTTPStatus,
) -> None:
    last_error: Exception | None = None
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError(f"process exited early with status {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == expected:
                    return
        except Exception as exc:  # noqa: BLE001 - startup retry diagnostics
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"HTTP endpoint did not become ready: {last_error!r}")


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(_post_raw(url, payload, accept="application/json"))


def _post_raw(url: str, payload: dict[str, Any], *, accept: str) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": accept, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


def _stream_text(body: str) -> str:
    parts = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        for choice in event.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                parts.append(content)
    return "".join(parts)


if __name__ == "__main__":
    unittest.main()
