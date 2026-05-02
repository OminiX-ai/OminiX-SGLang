#!/usr/bin/env python3
"""Cross-repo live check: OminiX-API router-only -> sglang-ominix v0 shim."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any


TOKEN = "bridge-check-token"


class OminiXApiToV0ShimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sglang_repo = Path(__file__).resolve().parents[2]
        self.ominix_api_repo = Path(__file__).resolve().parents[3] / "OminiX-API"
        self.api_binary = (
            self.ominix_api_repo / "target/debug/ominix-api-router-only"
        )
        if not self.api_binary.exists():
            self.skipTest(f"missing router-only binary: {self.api_binary}")

        self.scheduler_port = _free_port()
        self.api_port = _free_port()
        self.scheduler_proc = self._start_scheduler()
        self.addCleanup(_stop_process, self.scheduler_proc)
        self.api_proc = self._start_api()
        self.addCleanup(_stop_process, self.api_proc)
        self.base_url = f"http://127.0.0.1:{self.api_port}"
        _wait_for_http(f"{self.base_url}/health", self.api_proc)

    def test_openai_chat_completion_and_native_generate_route_to_shim(self):
        chat = _post_json(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "say pong"}],
                "max_tokens": 4,
                "stream": False,
            },
        )
        completion = _post_json(
            f"{self.base_url}/v1/completions",
            {
                "model": "deepseek-v4-flash",
                "prompt": "say pong",
                "max_tokens": 4,
                "stream": False,
            },
        )
        native = _post_json(
            f"{self.base_url}/generate",
            {
                "model": "deepseek-v4-flash",
                "text": "say pong",
                "stream": False,
            },
        )
        server_info = _get_json(f"{self.base_url}/server_info")
        health_generate = _get_json(f"{self.base_url}/health_generate")
        model_info = _get_json(f"{self.base_url}/get_model_info")
        loads = _get_json(f"{self.base_url}/get_loads")
        abort = _post_json(
            f"{self.base_url}/abort_request",
            {"request_id": "req-abort", "reason": "test"},
        )
        flush = _post_json(
            f"{self.base_url}/flush_cache",
            {"request_id": "req-flush", "model": "deepseek-v4-flash"},
        )

        self.assertEqual(chat["choices"][0]["message"]["content"], "pong")
        self.assertEqual(completion["choices"][0]["text"], "pong")
        self.assertEqual(native["text"], "pong")
        self.assertEqual(server_info["scheduler_backend"], "ominix-worker-v0-http")
        self.assertEqual(
            health_generate["scheduler_backend"], "ominix-worker-v0-http"
        )
        self.assertEqual(model_info["backend"], "sglang-ominix-v0-http-shim")
        self.assertTrue(model_info["is_generation"])
        self.assertEqual(loads["workers"][0]["backend"], "sglang-ominix-v0-http-shim")
        self.assertEqual(abort["aborted"], "req-abort")
        self.assertTrue(flush["flushed"])

    def test_openai_streaming_routes_to_shim(self):
        body = _post_raw(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "stream pong"}],
                "max_tokens": 4,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            accept="text/event-stream",
        )

        self.assertEqual(_stream_text(body), "pong")
        self.assertIn("data: [DONE]", body)

    def _start_scheduler(self) -> subprocess.Popen[str]:
        script = self.sglang_repo / "scripts/ominix/v0_http_scheduler_shim.py"
        proc = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.scheduler_port),
                "--token",
                TOKEN,
            ],
            cwd=self.sglang_repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        _wait_for_http(
            f"http://127.0.0.1:{self.scheduler_port}/health_generate",
            proc,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return proc

    def _start_api(self) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env.pop("DEEPSEEK_V4_ROUTER_ENABLE", None)
        env["PORT"] = str(self.api_port)
        env["RUST_LOG"] = "warn"
        env["OMINIX_V0_SCHEDULER_URL"] = (
            f"http://127.0.0.1:{self.scheduler_port}"
        )
        env["OMINIX_V0_SCHEDULER_TOKEN"] = TOKEN
        env["OMINIX_V0_SCHEDULER_TIMEOUT_SECS"] = "5"
        return subprocess.Popen(
            [str(self.api_binary)],
            cwd=self.ominix_api_repo,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )


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


def _wait_for_http(
    url: str,
    proc: subprocess.Popen[str],
    headers: dict[str, str] | None = None,
) -> None:
    last_error: Exception | None = None
    for _ in range(80):
        if proc.poll() is not None:
            raise RuntimeError(f"process exited early with status {proc.returncode}")
        try:
            request = urllib.request.Request(url, headers=headers or {}, method="GET")
            urllib.request.urlopen(request, timeout=0.5).read()
            return
        except Exception as exc:  # noqa: BLE001 - startup probe diagnostics
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
        headers={
            "Accept": accept,
            "Content-Type": "application/json",
        },
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
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                parts.append(content)
    return "".join(parts)


if __name__ == "__main__":
    unittest.main()
