#!/usr/bin/env python3
"""OminiX worker v0 HTTP/SSE shim for the scheduler-only SGLang fork.

This is an internal boundary adapter for OminiX-API. It is intentionally not an
OpenAI-compatible public server. The default fake backend emits deterministic
WorkerEvent SSE records through the same adapter that will later wrap real
SGLang GenerateResponse gRPC messages.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from worker_event_adapter import PROTOCOL_VERSION, adapt_generate_response_stream
from worker_token_boundary import (
    MissingTokenizerError,
    TokenBoundaryError,
    TokenDeltaDecoder,
    prepare_generate_request,
)


MESSAGE_TYPE_GENERATE_REQUEST = "GenerateRequest"
SHIM_BACKEND = "sglang-ominix-v0-http-shim"
DEFAULT_RESPONSE_TEXT = "pong"


@dataclass(frozen=True)
class ShimConfig:
    host: str
    port: int
    token: str | None
    generate_path: str
    response_text: str
    mode: str
    grpc_target: str | None
    grpc_timeout_secs: float
    tokenizer_path: str | None
    model_id: str


class OminiXV0SchedulerShim(ThreadingHTTPServer):
    config: ShimConfig

    def __init__(self, server_address: tuple[str, int], config: ShimConfig):
        super().__init__(server_address, OminiXV0SchedulerHandler)
        self.config = config
        self._tokenizer: Any | None = None

    def get_tokenizer(self) -> Any | None:
        if not self.config.tokenizer_path:
            return None
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.config.tokenizer_path)
        return self._tokenizer


class OminiXV0SchedulerHandler(BaseHTTPRequestHandler):
    server: OminiXV0SchedulerShim

    def do_GET(self) -> None:
        if not self._authorize():
            return

        path = self._request_path()

        if path == "/health_generate":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "scheduler_backend": SHIM_BACKEND,
                    "mode": self.server.config.mode,
                },
            )
            return

        if path == "/server_info":
            self._send_json(
                HTTPStatus.OK,
                {
                    "scheduler_backend": SHIM_BACKEND,
                    "mode": self.server.config.mode,
                    "protocol_version": PROTOCOL_VERSION,
                    "routes": {
                        "generate": self.server.config.generate_path,
                        "health_generate": "/health_generate",
                        "server_info": "/server_info",
                        "get_model_info": "/get_model_info",
                        "get_loads": "/get_loads",
                        "abort_request": "/abort_request",
                        "flush_cache": "/flush_cache",
                    },
                    "public_openai_api": False,
                },
            )
            return

        if path == "/get_model_info":
            try:
                grpc_response = self._grpc_control_get("GetModelInfo")
            except RuntimeError as exc:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": {"message": str(exc)}},
                )
                return
            if grpc_response is not None:
                self._send_json(HTTPStatus.OK, grpc_response)
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "backend": SHIM_BACKEND,
                    "mode": self.server.config.mode,
                    "model_path": self.server.config.model_id,
                    "tokenizer_path": self.server.config.tokenizer_path,
                    "is_generation": True,
                    "scheduler_grpc_target": self.server.config.grpc_target,
                },
            )
            return

        if path == "/get_loads":
            fallback_reason = None
            try:
                grpc_response = self._grpc_control_get("GetLoads")
            except RuntimeError as exc:
                fallback_reason = str(exc)
                grpc_response = None
            if grpc_response is not None:
                self._send_json(HTTPStatus.OK, grpc_response)
                return
            self._send_json(
                HTTPStatus.OK,
                _local_loads_payload(self.server.config, fallback_reason),
            )
            return

        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"error": {"message": f"unknown route: {path}"}},
        )

    def do_POST(self) -> None:
        if not self._authorize():
            return

        path = self._request_path()
        if path in {"/abort_request", "/flush_cache"}:
            try:
                request = self._read_json_body()
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": str(exc)}},
                )
                return
            try:
                response = self._control_response(path, request)
            except RuntimeError as exc:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": {"message": str(exc)}},
                )
                return
            self._send_json(HTTPStatus.OK, response)
            return

        if path != self.server.config.generate_path:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"message": f"unknown route: {path}"}},
            )
            return

        try:
            request = self._read_json_body()
            self._validate_generate_request(request)
        except ValueError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(exc)}},
            )
            return

        try:
            events = self._generate_worker_events(request)
        except ValueError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(exc)}},
            )
            return
        except RuntimeError as exc:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": {"message": str(exc)}},
            )
            return

        self._send_worker_event_sse(events)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(
            f"{self.address_string()} - - [{self.log_date_time_string()}] "
            + fmt % args,
            file=sys.stderr,
        )

    def _authorize(self) -> bool:
        token = self.server.config.token
        if not token:
            return True
        expected = f"Bearer {token}"
        if self.headers.get("Authorization") == expected:
            return True
        self._send_json(
            HTTPStatus.UNAUTHORIZED,
            {"error": {"message": "missing or invalid scheduler bearer token"}},
        )
        return False

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length") or "0")
        if content_length <= 0:
            raise ValueError("request body is empty")
        raw_body = self.rfile.read(content_length)
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("request body must be a JSON object")
        return decoded

    def _validate_generate_request(self, request: dict[str, Any]) -> None:
        protocol_version = request.get("protocol_version")
        if protocol_version is not None and protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol_version {protocol_version!r}; "
                f"expected {PROTOCOL_VERSION!r}"
            )

        message_type = request.get("message_type")
        if (
            message_type is not None
            and message_type != MESSAGE_TYPE_GENERATE_REQUEST
        ):
            raise ValueError(
                f"unsupported message_type {message_type!r}; "
                f"expected {MESSAGE_TYPE_GENERATE_REQUEST!r}"
            )

    def _generate_worker_events(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if self.server.config.mode == "fake":
            return self._fake_generate(request)
        if self.server.config.mode == "grpc":
            if not self.server.config.grpc_target:
                raise RuntimeError("--grpc-target is required when --mode grpc")
            tokenizer = (
                self.server.get_tokenizer()
                if _request_needs_tokenizer(request)
                else None
            )
            return _grpc_generate_worker_events(
                request,
                grpc_target=self.server.config.grpc_target,
                timeout_secs=self.server.config.grpc_timeout_secs,
                tokenizer=tokenizer,
            )
        raise RuntimeError(f"unsupported backend mode: {self.server.config.mode}")

    def _fake_generate(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        request_id = str(request.get("request_id") or f"shim-{uuid.uuid4()}")
        trace_id = _optional_str(request.get("trace_id"))
        prompt_tokens = _estimate_prompt_tokens(request)
        response_text = self.server.config.response_text
        first_text, second_text = _split_text_delta(response_text)
        token_ids = [31337, 31338]
        now_ms = int(time.time() * 1000)

        responses = [
            {
                "request_id": request_id,
                "chunk": {
                    "token_ids": [token_ids[0]],
                    "text_deltas": [first_text],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 1,
                    "index": 0,
                },
            },
            {
                "request_id": request_id,
                "complete": {
                    "output_ids": [token_ids[1]],
                    "text_deltas": [second_text],
                    "finish_reason": "stop",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 2,
                    "index": 0,
                },
            },
        ]

        yield from adapt_generate_response_stream(
            responses,
            trace_id=trace_id,
            request_id=request_id,
            created_at_ms=now_ms,
        )

    def _send_worker_event_sse(self, events: Iterator[dict[str, Any]]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        for event in events:
            payload = json.dumps(event, separators=(",", ":")).encode("utf-8")
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _request_path(self) -> str:
        return urlsplit(self.path).path

    def _control_response(self, path: str, request: dict[str, Any]) -> dict[str, Any]:
        if path == "/abort_request":
            grpc_response = self._grpc_abort(request)
            if grpc_response is not None:
                return grpc_response
            return {
                "ok": True,
                "backend": SHIM_BACKEND,
                "mode": self.server.config.mode,
                "aborted": request.get("request_id") or request.get("rid"),
                "reason": request.get("reason"),
            }
        return {
            "ok": True,
            "backend": SHIM_BACKEND,
            "mode": self.server.config.mode,
            "flushed": True,
            "request_id": request.get("request_id"),
        }

    def _grpc_control_get(self, method_name: str) -> dict[str, Any] | None:
        if self.server.config.mode != "grpc":
            return None
        if not self.server.config.grpc_target:
            raise RuntimeError("--grpc-target is required when --mode grpc")
        if method_name == "GetModelInfo":
            return _grpc_unary_json(
                self.server.config.grpc_target,
                self.server.config.grpc_timeout_secs,
                method_name,
                "GetModelInfoRequest",
            )
        if method_name == "GetLoads":
            return _grpc_unary_json(
                self.server.config.grpc_target,
                self.server.config.grpc_timeout_secs,
                method_name,
                "GetLoadsRequest",
            )
        raise RuntimeError(f"unsupported gRPC control method: {method_name}")

    def _grpc_abort(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if self.server.config.mode != "grpc":
            return None
        request_id = request.get("request_id") or request.get("rid")
        if not request_id:
            raise RuntimeError("abort_request requires request_id or rid")
        response = _grpc_unary_json(
            self.server.config.grpc_target,
            self.server.config.grpc_timeout_secs,
            "Abort",
            "AbortRequest",
            request_id=str(request_id),
            reason=str(request.get("reason") or ""),
        )
        response.setdefault("ok", bool(response.get("success", True)))
        response.setdefault("aborted", str(request_id))
        return response


def _estimate_prompt_tokens(request: dict[str, Any]) -> int:
    input_payload = request.get("input")
    if isinstance(input_payload, dict):
        if isinstance(input_payload.get("tokens"), list):
            return len(input_payload["tokens"])
        if isinstance(input_payload.get("messages"), list):
            return max(1, len(input_payload["messages"]) * 8)
        for key in ("prompt", "text"):
            value = input_payload.get(key)
            if isinstance(value, str):
                return max(1, len(value.split()))

    for key in ("prompt", "text"):
        value = request.get(key)
        if isinstance(value, str):
            return max(1, len(value.split()))

    return 1


def _local_loads_payload(
    config: ShimConfig,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    worker = {
        "worker_id": "sglang-ominix-shim-0",
        "backend": SHIM_BACKEND,
        "mode": config.mode,
        "model": config.model_id,
        "running_requests": 0,
        "waiting_requests": 0,
        "scheduler_grpc_target": config.grpc_target,
        "loads_source": "shim-local",
    }
    if fallback_reason:
        worker["loads_source"] = "shim-local-after-grpc-error"
        worker["grpc_error"] = fallback_reason
    return {"workers": [worker]}


def _grpc_generate_worker_events(
    request: dict[str, Any],
    *,
    grpc_target: str,
    timeout_secs: float,
    tokenizer: Any | None,
) -> Iterator[dict[str, Any]]:
    try:
        prepared = prepare_generate_request(request, tokenizer=tokenizer)
    except MissingTokenizerError as exc:
        raise ValueError(
            f"{exc}; pass --tokenizer-path for text/chat/completion requests"
        ) from exc
    except TokenBoundaryError as exc:
        raise ValueError(str(exc)) from exc

    request_id = prepared.request_id
    trace_id = _optional_str(request.get("trace_id"))
    grpc, message_to_dict, pb2, pb2_grpc = _load_grpc_modules()
    sglang_request = _build_sglang_generate_request(
        request_id=request_id,
        original_text=prepared.original_text,
        input_ids=prepared.input_ids,
        sampling=prepared.sampling_params,
        response_options=_response_options_payload(request),
        stream=prepared.stream,
        pb2=pb2,
    )
    channel = grpc.insecure_channel(
        grpc_target,
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )
    stub = pb2_grpc.SglangSchedulerStub(channel)
    try:
        response_stream = stub.Generate(sglang_request, timeout=timeout_secs)
    except Exception as exc:
        _close_grpc_channel(channel)
        raise RuntimeError(f"SGLang Generate gRPC call failed: {exc}") from exc

    text_injector = (
        GrpcTextDeltaInjector(
            tokenizer,
            sampling=prepared.sampling_params,
            response_options=_response_options_payload(request),
        )
        if tokenizer is not None
        else None
    )
    return adapt_generate_response_stream(
        _generate_response_dicts(
            response_stream,
            message_to_dict=message_to_dict,
            close_channel=lambda: _close_grpc_channel(channel),
            text_injector=text_injector,
        ),
        trace_id=trace_id,
        request_id=request_id,
        created_at_ms=int(time.time() * 1000),
    )


def _load_grpc_modules() -> tuple[Any, Any, Any, Any]:
    try:
        grpc = importlib.import_module("grpc")
        json_format = importlib.import_module("google.protobuf.json_format")
        pb2 = _import_first(
            (
                "smg_grpc_proto.sglang_scheduler_pb2",
                "sglang.srt.grpc.sglang_scheduler_pb2",
            )
        )
        pb2_grpc = _import_first(
            (
                "smg_grpc_proto.sglang_scheduler_pb2_grpc",
                "sglang.srt.grpc.sglang_scheduler_pb2_grpc",
            )
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "grpc mode requires installed modules: "
            "grpc, google.protobuf.json_format, and either smg_grpc_proto "
            "or sglang.srt.grpc generated scheduler modules; missing "
            f"{exc.name or exc}"
        ) from exc

    return grpc, json_format.MessageToDict, pb2, pb2_grpc


def _import_first(module_names: tuple[str, ...]) -> Any:
    last_error: ModuleNotFoundError | None = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError(module_names[0])


def _grpc_unary_json(
    grpc_target: str | None,
    timeout_secs: float,
    method_name: str,
    request_class_name: str,
    **request_fields: Any,
) -> dict[str, Any]:
    if not grpc_target:
        raise RuntimeError("--grpc-target is required when --mode grpc")

    grpc, message_to_dict, pb2, pb2_grpc = _load_grpc_modules()
    request_cls = getattr(pb2, request_class_name)
    request = request_cls(**request_fields)
    channel = grpc.insecure_channel(
        grpc_target,
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )
    try:
        stub = pb2_grpc.SglangSchedulerStub(channel)
        method = getattr(stub, method_name)
        response = method(request, timeout=timeout_secs)
        converted = message_to_dict(response, preserving_proto_field_name=True)
        return converted if isinstance(converted, dict) else {}
    except Exception as exc:
        raise RuntimeError(f"SGLang {method_name} gRPC call failed: {exc}") from exc
    finally:
        _close_grpc_channel(channel)


def _build_sglang_generate_request(
    *,
    request_id: str,
    original_text: str,
    input_ids: list[int],
    sampling: dict[str, Any],
    response_options: dict[str, Any],
    stream: bool,
    pb2: Any,
) -> Any:
    return pb2.GenerateRequest(
        request_id=request_id,
        tokenized=pb2.TokenizedInput(
            input_ids=input_ids,
            original_text=original_text or "tokenized input",
        ),
        sampling_params=pb2.SamplingParams(
            max_new_tokens=_coerce_optional_int(
                _first_present(
                    sampling,
                    "max_new_tokens",
                    "max_tokens",
                    "max_completion_tokens",
                ),
                default=8,
            ),
            temperature=_coerce_optional_float(sampling.get("temperature"), default=0.0),
            top_p=_coerce_optional_float(sampling.get("top_p"), default=1.0),
            top_k=_coerce_optional_int(sampling.get("top_k"), default=-1),
            min_p=_coerce_optional_float(sampling.get("min_p"), default=0.0),
            frequency_penalty=_coerce_optional_float(
                sampling.get("frequency_penalty"),
                default=0.0,
            ),
            presence_penalty=_coerce_optional_float(
                sampling.get("presence_penalty"),
                default=0.0,
            ),
            repetition_penalty=_coerce_optional_float(
                sampling.get("repetition_penalty"),
                default=1.0,
            ),
            stop=_coerce_str_list(sampling.get("stop")),
            stop_token_ids=_coerce_int_list(sampling.get("stop_token_ids")),
            n=_coerce_optional_int(sampling.get("n"), default=1),
            skip_special_tokens=bool(response_options.get("skip_special_tokens", True)),
            spaces_between_special_tokens=bool(
                response_options.get("spaces_between_special_tokens", True)
            ),
        ),
        stream=stream,
    )


def _generate_response_dicts(
    responses: Iterator[Any],
    *,
    message_to_dict: Any,
    close_channel: Any,
    text_injector: "GrpcTextDeltaInjector | None",
) -> Iterator[dict[str, Any]]:
    try:
        for response in responses:
            converted = message_to_dict(response, preserving_proto_field_name=True)
            if not isinstance(converted, dict):
                raise RuntimeError("SGLang GenerateResponse did not convert to a dict")
            if text_injector is not None:
                converted = text_injector.apply(converted)
            yield converted
    finally:
        close_channel()


class GrpcTextDeltaInjector:
    def __init__(
        self,
        tokenizer: Any,
        *,
        sampling: dict[str, Any],
        response_options: dict[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.sampling = sampling
        self.response_options = response_options
        self.decoders: dict[int, TokenDeltaDecoder] = {}
        self.seen_counts: dict[int, int] = {}
        self.seen_token_ids: dict[int, list[int]] = {}

    def apply(self, response: dict[str, Any]) -> dict[str, Any]:
        if isinstance(response.get("chunk"), dict):
            chunk = dict(response["chunk"])
            self._inject_chunk_text(chunk)
            response = {**response, "chunk": chunk}
        elif isinstance(response.get("complete"), dict):
            complete = dict(response["complete"])
            self._inject_complete_text(complete)
            response = {**response, "complete": complete}
        return response

    def _inject_chunk_text(self, chunk: dict[str, Any]) -> None:
        sequence_index = _coerce_optional_int(chunk.get("index"), default=0)
        token_ids = _coerce_int_list(chunk.get("token_ids"))
        if not token_ids:
            return
        decoder = self._decoder(sequence_index)
        deltas = decoder.accept(token_ids)
        self.seen_counts[sequence_index] = (
            self.seen_counts.get(sequence_index, 0) + len(token_ids)
        )
        self.seen_token_ids.setdefault(sequence_index, []).extend(token_ids)
        if deltas:
            chunk["text_deltas"] = deltas

    def _inject_complete_text(self, complete: dict[str, Any]) -> None:
        sequence_index = _coerce_optional_int(complete.get("index"), default=0)
        output_ids = _coerce_int_list(complete.get("output_ids"))
        seen_ids = self.seen_token_ids.get(sequence_index, [])
        new_ids: list[int] = []
        if output_ids:
            if not seen_ids:
                new_ids = output_ids
            elif output_ids[: len(seen_ids)] == seen_ids:
                new_ids = output_ids[len(seen_ids) :]
                complete["output_ids"] = new_ids
            else:
                new_ids = output_ids

        decoder = self._decoder(sequence_index)
        deltas = decoder.accept(new_ids) if new_ids else []
        final_delta = decoder.finish()
        if final_delta:
            deltas.append(final_delta)
        if deltas:
            complete["text_deltas"] = deltas

    def _decoder(self, sequence_index: int) -> TokenDeltaDecoder:
        decoder = self.decoders.get(sequence_index)
        if decoder is None:
            decoder = TokenDeltaDecoder(
                self.tokenizer,
                stop_strings=_coerce_str_list(self.sampling.get("stop")),
                stop_token_ids=_coerce_int_list(self.sampling.get("stop_token_ids")),
                skip_special_tokens=bool(
                    self.response_options.get("skip_special_tokens", True)
                ),
                spaces_between_special_tokens=bool(
                    self.response_options.get("spaces_between_special_tokens", True)
                ),
            )
            self.decoders[sequence_index] = decoder
        return decoder


def _request_needs_tokenizer(request: dict[str, Any]) -> bool:
    input_payload = request.get("input")
    if isinstance(input_payload, dict):
        kind = input_payload.get("kind")
        if kind in {"chat", "completion", "responses", "text"}:
            return True
        return any(
            key in input_payload for key in ("messages", "prompt", "text")
        )
    return any(key in request for key in ("prompt", "text"))


def _load_tokenizer(tokenizer_path: str) -> Any:
    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "text/chat grpc mode requires transformers; install transformers "
            "or use tokenized input"
        ) from exc

    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_tokenizer is None:
        raise RuntimeError("transformers.AutoTokenizer is not available")

    try:
        return auto_tokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load tokenizer from {tokenizer_path}: {exc}") from exc


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _response_options_payload(request: dict[str, Any]) -> dict[str, Any]:
    response_options = request.get("response_options")
    return response_options if isinstance(response_options, dict) else {}


def _coerce_optional_int(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _coerce_optional_float(value: Any, *, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("sampling.stop must be a string or string list")
    return [str(item) for item in value]


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("expected an integer list")
    return [int(item) for item in value]


def _close_grpc_channel(channel: Any) -> None:
    close = getattr(channel, "close", None)
    if close is not None:
        close()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _split_text_delta(text: str) -> tuple[str, str]:
    if len(text) <= 1:
        return text, ""
    split_at = max(1, len(text) // 2)
    return text[:split_at], text[split_at:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19091)
    parser.add_argument("--token")
    parser.add_argument("--generate-path", default="/generate")
    parser.add_argument("--response-text", default=DEFAULT_RESPONSE_TEXT)
    parser.add_argument(
        "--mode",
        choices=("fake", "grpc"),
        default="fake",
        help="Backend mode. Use grpc to bridge to SglangScheduler.Generate.",
    )
    parser.add_argument(
        "--grpc-target",
        help="SGLang scheduler gRPC target host:port, required for --mode grpc.",
    )
    parser.add_argument("--grpc-timeout-secs", type=float, default=120.0)
    parser.add_argument(
        "--tokenizer-path",
        help="Tokenizer path/model id for grpc text/chat/completion requests.",
    )
    parser.add_argument(
        "--model-id",
        default="unknown",
        help="Model identifier reported by scheduler control routes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generate_path = args.generate_path
    if not generate_path.startswith("/"):
        generate_path = f"/{generate_path}"

    config = ShimConfig(
        host=args.host,
        port=args.port,
        token=args.token,
        generate_path=generate_path,
        response_text=args.response_text,
        mode=args.mode,
        grpc_target=args.grpc_target,
        grpc_timeout_secs=args.grpc_timeout_secs,
        tokenizer_path=args.tokenizer_path,
        model_id=args.model_id,
    )
    server = OminiXV0SchedulerShim((config.host, config.port), config)
    print(
        f"{SHIM_BACKEND} listening on http://{config.host}:{config.port}"
        f"{config.generate_path}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
