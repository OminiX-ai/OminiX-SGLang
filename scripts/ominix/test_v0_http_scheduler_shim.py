#!/usr/bin/env python3
"""Live checks for the OminiX worker v0 HTTP scheduler shim."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

import v0_http_scheduler_shim as shim


TOKEN = "shim-test-token"


class V0HttpSchedulerShimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.port = _free_port()
        script = Path(__file__).with_name("v0_http_scheduler_shim.py")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--token",
                TOKEN,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.addCleanup(_stop_process, self.proc)
        self.base_url = f"http://127.0.0.1:{self.port}"
        _wait_for_health(self.base_url)

    def test_health_and_server_info(self):
        health = _json_get(f"{self.base_url}/health_generate")
        info = _json_get(f"{self.base_url}/server_info")

        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["scheduler_backend"], "sglang-ominix-v0-http-shim")
        self.assertEqual(info["routes"]["generate"], "/generate")
        self.assertEqual(info["routes"]["get_model_info"], "/get_model_info")
        self.assertEqual(info["routes"]["get_loads"], "/get_loads")
        self.assertFalse(info["public_openai_api"])

    def test_control_routes_return_scheduler_metadata(self):
        model_info = _json_get(f"{self.base_url}/get_model_info")
        loads = _json_get(f"{self.base_url}/get_loads")
        abort = _json_post(
            f"{self.base_url}/abort_request",
            {"request_id": "req-abort", "reason": "test"},
        )
        flush = _json_post(
            f"{self.base_url}/flush_cache",
            {"request_id": "req-flush"},
        )

        self.assertEqual(model_info["backend"], "sglang-ominix-v0-http-shim")
        self.assertTrue(model_info["is_generation"])
        self.assertEqual(loads["workers"][0]["backend"], "sglang-ominix-v0-http-shim")
        self.assertEqual(abort["aborted"], "req-abort")
        self.assertEqual(abort["reason"], "test")
        self.assertTrue(flush["flushed"])

    def test_bearer_token_is_required(self):
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=json.dumps({"request_id": "unauthorized"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=5)

        self.assertEqual(raised.exception.code, 401)

    def test_generate_request_returns_worker_event_sse_with_text_deltas(self):
        events = _post_generate(
            self.base_url,
            {
                "protocol_version": "ominix.worker.v0",
                "message_type": "GenerateRequest",
                "request_id": "req-shim-1",
                "trace_id": "trace-shim-1",
                "input": {
                    "kind": "text",
                    "prompt": "ping",
                },
                "sampling": {
                    "max_tokens": 2,
                    "temperature": 0.0,
                },
                "stream": False,
            },
        )

        self.assertEqual(
            [event["kind"] for event in events],
            ["prefill_done", "token", "token", "usage", "done"],
        )
        self.assertEqual(
            "".join(event.get("text_delta", "") for event in events), "pong"
        )
        self.assertTrue(all(event["request_id"] == "req-shim-1" for event in events))
        self.assertTrue(all(event.get("trace_id") == "trace-shim-1" for event in events))

    def test_native_generate_shape_is_accepted(self):
        events = _post_generate(
            self.base_url,
            {
                "request_id": "req-native",
                "prompt": "ping",
                "sampling_params": {
                    "max_new_tokens": 2,
                    "temperature": 0.0,
                },
            },
        )

        self.assertEqual(events[-1]["kind"], "done")
        self.assertEqual("".join(event.get("text_delta", "") for event in events), "pong")

    def test_grpc_mode_rejects_text_input_until_tokenizer_slice_lands(self):
        port = _free_port()
        proc = _start_shim(
            port,
            "--mode",
            "grpc",
            "--grpc-target",
            "127.0.0.1:1",
        )
        self.addCleanup(_stop_process, proc)
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_health(base_url)

        error = _post_generate_error(
            base_url,
            {
                "protocol_version": "ominix.worker.v0",
                "message_type": "GenerateRequest",
                "request_id": "req-grpc-text",
                "input": {
                    "kind": "chat",
                    "messages": [{"role": "user", "content": "Say pong."}],
                },
                "sampling": {"max_tokens": 1},
                "stream": True,
            },
        )

        self.assertEqual(error.code, 400)
        body = error.read().decode("utf-8")
        self.assertIn("tokenizer", body)
        self.assertIn("--tokenizer-path", body)

    def test_grpc_mode_bearer_token_is_required(self):
        port = _free_port()
        proc = _start_shim(
            port,
            "--mode",
            "grpc",
            "--grpc-target",
            "127.0.0.1:1",
        )
        self.addCleanup(_stop_process, proc)
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_health(base_url)

        request = urllib.request.Request(
            f"{base_url}/generate",
            data=json.dumps({"request_id": "unauthorized"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=5)

        self.assertEqual(raised.exception.code, 401)


class V0GrpcBridgeUnitTest(unittest.TestCase):
    def test_grpc_mode_maps_token_input_and_adapts_fake_protobuf_responses(self):
        fake_grpc = FakeGrpcModule()
        fake_pb2 = FakePb2Module()
        fake_json_format = types.ModuleType("google.protobuf.json_format")
        fake_json_format.MessageToDict = fake_message_to_dict
        fake_pb2_grpc = types.ModuleType(
            "smg_grpc_proto.sglang_scheduler_pb2_grpc"
        )
        fake_pb2_grpc.SglangSchedulerStub = FakeSglangSchedulerStub

        modules = {
            "grpc": fake_grpc,
            "google.protobuf.json_format": fake_json_format,
            "smg_grpc_proto": types.ModuleType("smg_grpc_proto"),
            "smg_grpc_proto.sglang_scheduler_pb2": fake_pb2,
            "smg_grpc_proto.sglang_scheduler_pb2_grpc": fake_pb2_grpc,
        }
        payload = {
            "protocol_version": "ominix.worker.v0",
            "message_type": "GenerateRequest",
            "request_id": "req-grpc-tokens",
            "trace_id": "trace-grpc-tokens",
            "input": {"kind": "tokens", "tokens": [11, 22, 33]},
            "sampling": {
                "max_tokens": 4,
                "temperature": 0.25,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.1,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.1,
                "stop": ["stop"],
                "stop_token_ids": [99],
                "n": 1,
            },
            "stream": True,
        }

        with patch.dict(sys.modules, modules):
            events = list(
                shim._grpc_generate_worker_events(
                    payload,
                    grpc_target="127.0.0.1:30000",
                    timeout_secs=5.0,
                    tokenizer=None,
                )
            )

        request = FakeSglangSchedulerStub.last_request
        self.assertIsNotNone(request)
        self.assertEqual(request.request_id, "req-grpc-tokens")
        self.assertEqual(request.tokenized.input_ids, [11, 22, 33])
        self.assertEqual(request.sampling_params.max_new_tokens, 4)
        self.assertEqual(request.sampling_params.temperature, 0.25)
        self.assertEqual(request.sampling_params.top_p, 0.9)
        self.assertEqual(request.sampling_params.top_k, 20)
        self.assertEqual(request.sampling_params.min_p, 0.1)
        self.assertEqual(request.sampling_params.frequency_penalty, 0.2)
        self.assertEqual(request.sampling_params.presence_penalty, 0.3)
        self.assertEqual(request.sampling_params.repetition_penalty, 1.1)
        self.assertEqual(request.sampling_params.stop, ["stop"])
        self.assertEqual(request.sampling_params.stop_token_ids, [99])
        self.assertTrue(request.stream)
        self.assertEqual(fake_grpc.channels[-1].target, "127.0.0.1:30000")
        self.assertTrue(fake_grpc.channels[-1].closed)

        self.assertEqual(
            [event["kind"] for event in events],
            ["prefill_done", "token", "token", "usage", "done"],
        )
        self.assertEqual(
            "".join(event.get("text_delta", "") for event in events), "OK"
        )
        self.assertTrue(
            all(event["request_id"] == "req-grpc-tokens" for event in events)
        )
        self.assertTrue(
            all(event.get("trace_id") == "trace-grpc-tokens" for event in events)
        )

    def test_grpc_mode_accepts_transitional_native_input_ids(self):
        fake_grpc = FakeGrpcModule()
        fake_pb2 = FakePb2Module()
        fake_json_format = types.ModuleType("google.protobuf.json_format")
        fake_json_format.MessageToDict = fake_message_to_dict
        fake_pb2_grpc = types.ModuleType(
            "smg_grpc_proto.sglang_scheduler_pb2_grpc"
        )
        fake_pb2_grpc.SglangSchedulerStub = FakeSglangSchedulerStub
        modules = {
            "grpc": fake_grpc,
            "google.protobuf.json_format": fake_json_format,
            "smg_grpc_proto": types.ModuleType("smg_grpc_proto"),
            "smg_grpc_proto.sglang_scheduler_pb2": fake_pb2,
            "smg_grpc_proto.sglang_scheduler_pb2_grpc": fake_pb2_grpc,
        }

        with patch.dict(sys.modules, modules):
            list(
                shim._grpc_generate_worker_events(
                    {
                        "request_id": "req-native-input-ids",
                        "input_ids": [44, 55],
                        "sampling_params": {"max_new_tokens": 2},
                    },
                    grpc_target="127.0.0.1:30000",
                    timeout_secs=5.0,
                    tokenizer=None,
                )
            )

        request = FakeSglangSchedulerStub.last_request
        self.assertIsNotNone(request)
        self.assertEqual(request.tokenized.input_ids, [44, 55])
        self.assertEqual(request.sampling_params.max_new_tokens, 2)

    def test_grpc_mode_uses_injected_tokenizer_for_text_input(self):
        fake_grpc = FakeGrpcModule()
        fake_pb2 = FakePb2Module()
        fake_json_format = types.ModuleType("google.protobuf.json_format")
        fake_json_format.MessageToDict = fake_message_to_dict
        fake_pb2_grpc = types.ModuleType(
            "smg_grpc_proto.sglang_scheduler_pb2_grpc"
        )
        fake_pb2_grpc.SglangSchedulerStub = FakeSglangSchedulerStub
        modules = {
            "grpc": fake_grpc,
            "google.protobuf.json_format": fake_json_format,
            "smg_grpc_proto": types.ModuleType("smg_grpc_proto"),
            "smg_grpc_proto.sglang_scheduler_pb2": fake_pb2,
            "smg_grpc_proto.sglang_scheduler_pb2_grpc": fake_pb2_grpc,
        }

        with patch.dict(sys.modules, modules):
            list(
                shim._grpc_generate_worker_events(
                    {
                        "request_id": "req-text-input",
                        "input": {"kind": "completion", "prompt": "ping pong"},
                        "sampling": {"max_tokens": 3},
                        "stream": True,
                    },
                    grpc_target="127.0.0.1:30000",
                    timeout_secs=5.0,
                    tokenizer=DecodeStubTokenizer(),
                )
            )

        request = FakeSglangSchedulerStub.last_request
        self.assertIsNotNone(request)
        self.assertEqual(request.tokenized.input_ids, [901, 902])
        self.assertEqual(request.tokenized.original_text, "ping pong")
        self.assertEqual(request.sampling_params.max_new_tokens, 3)

    def test_grpc_module_loader_accepts_sglang_native_proto_package(self):
        fake_grpc = FakeGrpcModule()
        fake_pb2 = FakePb2Module()
        fake_json_format = types.ModuleType("google.protobuf.json_format")
        fake_json_format.MessageToDict = fake_message_to_dict
        fake_pb2_grpc = types.ModuleType(
            "sglang.srt.grpc.sglang_scheduler_pb2_grpc"
        )
        fake_pb2_grpc.SglangSchedulerStub = FakeSglangSchedulerStub
        modules = {
            "grpc": fake_grpc,
            "google.protobuf.json_format": fake_json_format,
            "sglang": types.ModuleType("sglang"),
            "sglang.srt": types.ModuleType("sglang.srt"),
            "sglang.srt.grpc": types.ModuleType("sglang.srt.grpc"),
            "sglang.srt.grpc.sglang_scheduler_pb2": fake_pb2,
            "sglang.srt.grpc.sglang_scheduler_pb2_grpc": fake_pb2_grpc,
        }

        with patch.dict(sys.modules, modules):
            _, _, pb2, pb2_grpc = shim._load_grpc_modules()

        self.assertIs(pb2, fake_pb2)
        self.assertIs(pb2_grpc, fake_pb2_grpc)

    def test_grpc_text_delta_injector_decodes_real_token_id_shape(self):
        injector = shim.GrpcTextDeltaInjector(
            DecodeStubTokenizer(),
            sampling={},
            response_options={},
        )
        first = injector.apply(
            {
                "request_id": "req-decode",
                "chunk": {
                    "token_ids": [901],
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "index": 0,
                },
            }
        )
        second = injector.apply(
            {
                "request_id": "req-decode",
                "complete": {
                    "output_ids": [901, 902],
                    "finish_reason": "stop",
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                    "index": 0,
                },
            }
        )

        events = list(
            shim.adapt_generate_response_stream(
                [first, second],
                request_id="req-decode",
            )
        )

        self.assertEqual("".join(event.get("text_delta", "") for event in events), "OK")

    def test_grpc_control_unary_calls_native_scheduler_stub(self):
        fake_grpc = FakeGrpcModule()
        fake_pb2 = FakePb2Module()
        fake_json_format = types.ModuleType("google.protobuf.json_format")
        fake_json_format.MessageToDict = fake_message_to_dict
        fake_pb2_grpc = types.ModuleType(
            "sglang.srt.grpc.sglang_scheduler_pb2_grpc"
        )
        fake_pb2_grpc.SglangSchedulerStub = FakeSglangSchedulerStub
        modules = {
            "grpc": fake_grpc,
            "google.protobuf.json_format": fake_json_format,
            "sglang": types.ModuleType("sglang"),
            "sglang.srt": types.ModuleType("sglang.srt"),
            "sglang.srt.grpc": types.ModuleType("sglang.srt.grpc"),
            "sglang.srt.grpc.sglang_scheduler_pb2": fake_pb2,
            "sglang.srt.grpc.sglang_scheduler_pb2_grpc": fake_pb2_grpc,
        }

        with patch.dict(sys.modules, modules):
            response = shim._grpc_unary_json(
                "127.0.0.1:30000",
                5.0,
                "GetModelInfo",
                "GetModelInfoRequest",
            )

        self.assertEqual(response["backend"], "fake-grpc")
        self.assertTrue(fake_grpc.channels[-1].closed)

    def test_local_loads_payload_marks_grpc_fallback(self):
        payload = shim._local_loads_payload(
            shim.ShimConfig(
                host="127.0.0.1",
                port=19091,
                token=None,
                generate_path="/generate",
                response_text="pong",
                mode="grpc",
                grpc_target="127.0.0.1:30000",
                grpc_timeout_secs=5.0,
                tokenizer_path="/models/deepseek",
                model_id="deepseek-v4-flash",
            ),
            "deadline exceeded",
        )

        worker = payload["workers"][0]
        self.assertEqual(worker["loads_source"], "shim-local-after-grpc-error")
        self.assertEqual(worker["scheduler_grpc_target"], "127.0.0.1:30000")
        self.assertIn("deadline exceeded", worker["grpc_error"])


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


def _start_shim(port: int, *extra_args: str) -> subprocess.Popen[str]:
    script = Path(__file__).with_name("v0_http_scheduler_shim.py")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--token",
            TOKEN,
            *extra_args,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _wait_for_health(base_url: str) -> None:
    last_error: Exception | None = None
    for _ in range(50):
        try:
            _json_get(f"{base_url}/health_generate")
            return
        except Exception as exc:  # noqa: BLE001 - test startup retry path
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"shim did not become ready: {last_error!r}")


def _json_get(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_auth_headers(), method="GET")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_generate(base_url: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **_auth_headers(),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read().decode("utf-8")
    return _parse_sse_events(body)


def _json_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **_auth_headers(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_generate_error(
    base_url: str,
    payload: dict[str, Any],
) -> urllib.error.HTTPError:
    request = urllib.request.Request(
        f"{base_url}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **_auth_headers(),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc:
        return exc
    raise AssertionError("expected HTTPError")


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    events = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))
    return events


class FakeChannel:
    def __init__(self, target: str):
        self.target = target
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeGrpcModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("grpc")
        self.channels: list[FakeChannel] = []

    def insecure_channel(self, target: str, options: list[tuple[str, int]]):
        del options
        channel = FakeChannel(target)
        self.channels.append(channel)
        return channel


class FakeMessage:
    def __init__(self, **fields: Any):
        self.__dict__.update(fields)


class FakePb2Module(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("smg_grpc_proto.sglang_scheduler_pb2")
        self.GenerateRequest = FakeMessage
        self.TokenizedInput = FakeMessage
        self.SamplingParams = FakeMessage
        self.GetModelInfoRequest = FakeMessage
        self.GetLoadsRequest = FakeMessage
        self.AbortRequest = FakeMessage


class FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload


class FakeSglangSchedulerStub:
    last_request: FakeMessage | None = None

    def __init__(self, channel: FakeChannel):
        self.channel = channel

    def Generate(self, request: FakeMessage, timeout: float):
        self.__class__.last_request = request
        self.last_timeout = timeout
        return iter(
            [
                FakeResponse(
                    {
                        "request_id": request.request_id,
                        "chunk": {
                            "token_ids": [7001],
                            "text_deltas": ["O"],
                            "prompt_tokens": 3,
                            "completion_tokens": 1,
                            "index": 0,
                        },
                    }
                ),
                FakeResponse(
                    {
                        "request_id": request.request_id,
                        "complete": {
                            "output_ids": [7001, 7002],
                            "text_deltas": ["O", "K"],
                            "finish_reason": "stop",
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "index": 0,
                        },
                    }
                ),
            ]
        )

    def GetModelInfo(self, request: FakeMessage, timeout: float):
        del request, timeout
        return FakeResponse(
            {
                "backend": "fake-grpc",
                "model_path": "deepseek-v4-flash",
                "is_generation": True,
            }
        )

    def GetLoads(self, request: FakeMessage, timeout: float):
        del request, timeout
        return FakeResponse({"workers": [{"backend": "fake-grpc"}]})

    def Abort(self, request: FakeMessage, timeout: float):
        del timeout
        return FakeResponse({"success": True, "message": request.request_id})


def fake_message_to_dict(
    response: FakeResponse,
    *,
    preserving_proto_field_name: bool,
) -> dict[str, Any]:
    if not preserving_proto_field_name:
        raise AssertionError("expected preserving_proto_field_name=True")
    return response.payload


class StubTokenizer:
    def encode(self, text: str) -> list[int]:
        vocab = {"ping": 901, "pong": 902}
        return [vocab[word] for word in text.split()]


class DecodeStubTokenizer(StubTokenizer):
    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        vocab = {901: "O", 902: "K", 7001: "O", 7002: "K"}
        return "".join(vocab[token_id] for token_id in token_ids)


if __name__ == "__main__":
    unittest.main()
