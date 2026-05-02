#!/usr/bin/env python3
"""Pure-Python SGLang GenerateResponse dict to OminiX WorkerEvent dict adapter.

This module intentionally does not import grpc, protobuf, SGLang, or generated
proto modules. It accepts dicts like google.protobuf.json_format.MessageToDict
would produce for the current SGLang GenerateResponse shape with
preserving_proto_field_name=True.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


PROTOCOL_VERSION = "ominix.worker.v0"
MESSAGE_TYPE = "WorkerEvent"
_ALLOWED_FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "abort",
    "error",
}
_RETRYABLE_HTTP_CODES = {"408", "429", "500", "502", "503", "504"}


@dataclass
class WorkerEventAdapterState:
    """Per-stream adapter state."""

    saw_prefill: set[int] = field(default_factory=set)
    emitted_token_counts: dict[int, int] = field(default_factory=dict)
    last_usage: dict[int, dict[str, int]] = field(default_factory=dict)
    event_count: int = 0


def adapt_generate_response_stream(
    responses: Iterable[dict[str, Any]],
    *,
    trace_id: str | None = None,
    request_id: str | None = None,
    created_at_ms: int | Callable[[], int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield WorkerEvent-like dicts for a stream of GenerateResponse-like dicts."""

    state = WorkerEventAdapterState()
    for response in responses:
        yield from adapt_generate_response(
            response,
            state=state,
            trace_id=trace_id,
            request_id=request_id,
            created_at_ms=created_at_ms,
        )


def adapt_generate_response(
    response: dict[str, Any],
    *,
    state: WorkerEventAdapterState | None = None,
    trace_id: str | None = None,
    request_id: str | None = None,
    created_at_ms: int | Callable[[], int] | None = None,
) -> list[dict[str, Any]]:
    """Convert one GenerateResponse-like dict into zero or more WorkerEvents."""

    if state is None:
        state = WorkerEventAdapterState()

    rid = str(response.get("request_id") or request_id or "")
    if not rid:
        raise ValueError("GenerateResponse dict must include request_id")

    if _has_payload(response, "error"):
        error = response.get("error") or {}
        return [
            _base_event(
                "error",
                rid,
                state,
                trace_id=trace_id,
                created_at_ms=created_at_ms,
                finish_reason="error",
                error=_map_error(error),
            )
        ]

    if _has_payload(response, "chunk"):
        return _adapt_chunk(
            rid,
            response.get("chunk") or {},
            state,
            trace_id=trace_id,
            created_at_ms=created_at_ms,
        )

    if _has_payload(response, "complete"):
        return _adapt_complete(
            rid,
            response.get("complete") or {},
            state,
            trace_id=trace_id,
            created_at_ms=created_at_ms,
        )

    return []


def _adapt_chunk(
    request_id: str,
    chunk: dict[str, Any],
    state: WorkerEventAdapterState,
    *,
    trace_id: str | None,
    created_at_ms: int | Callable[[], int] | None,
) -> list[dict[str, Any]]:
    sequence_index = _coerce_int(chunk.get("index"), default=0)
    usage = _build_usage(chunk)
    events: list[dict[str, Any]] = []

    if sequence_index not in state.saw_prefill and usage["prompt_tokens"] > 0:
        state.saw_prefill.add(sequence_index)
        events.append(
            _base_event(
                "prefill_done",
                request_id,
                state,
                trace_id=trace_id,
                created_at_ms=created_at_ms,
                sequence_index=sequence_index,
                usage={
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": 0,
                    "total_tokens": usage["prompt_tokens"],
                    **_cached_tokens(usage),
                },
            )
        )

    token_ids = _coerce_int_list(chunk.get("token_ids"))
    output_logprobs = chunk.get("output_logprobs") or {}
    for position, token_id in enumerate(token_ids):
        event = _base_event(
            "token",
            request_id,
            state,
            trace_id=trace_id,
            created_at_ms=created_at_ms,
            token_id=token_id,
            sequence_index=sequence_index,
        )
        text_delta = _text_delta_for_position(chunk, position)
        if text_delta is not None:
            event["text_delta"] = text_delta
        logprobs = _map_output_logprobs(output_logprobs, position)
        if logprobs:
            event["logprobs"] = logprobs
        events.append(event)

    if token_ids:
        state.emitted_token_counts[sequence_index] = (
            state.emitted_token_counts.get(sequence_index, 0) + len(token_ids)
        )
        state.last_usage[sequence_index] = usage
    elif _has_usage(chunk):
        events.append(
            _base_event(
                "usage",
                request_id,
                state,
                trace_id=trace_id,
                created_at_ms=created_at_ms,
                sequence_index=sequence_index,
                usage=usage,
            )
        )
        state.last_usage[sequence_index] = usage

    return events


def _adapt_complete(
    request_id: str,
    complete: dict[str, Any],
    state: WorkerEventAdapterState,
    *,
    trace_id: str | None,
    created_at_ms: int | Callable[[], int] | None,
) -> list[dict[str, Any]]:
    sequence_index = _coerce_int(complete.get("index"), default=0)
    usage = _build_usage(complete)
    output_ids = _coerce_int_list(complete.get("output_ids"))
    emitted_count = state.emitted_token_counts.get(sequence_index, 0)
    token_ids_to_emit = _terminal_token_delta(output_ids, emitted_count)
    events: list[dict[str, Any]] = []

    output_logprobs = complete.get("output_logprobs") or {}
    logprob_offset = max(0, len(output_ids) - len(token_ids_to_emit))
    for position, token_id in enumerate(token_ids_to_emit):
        event = _base_event(
            "token",
            request_id,
            state,
            trace_id=trace_id,
            created_at_ms=created_at_ms,
            token_id=token_id,
            sequence_index=sequence_index,
        )
        text_delta = _text_delta_for_position(
            complete,
            logprob_offset + position,
            allow_full_text=emitted_count == 0,
        )
        if text_delta is not None:
            event["text_delta"] = text_delta
        logprobs = _map_output_logprobs(output_logprobs, logprob_offset + position)
        if logprobs:
            event["logprobs"] = logprobs
        events.append(event)

    if token_ids_to_emit:
        state.emitted_token_counts[sequence_index] = emitted_count + len(
            token_ids_to_emit
        )

    if _has_usage(complete):
        events.append(
            _base_event(
                "usage",
                request_id,
                state,
                trace_id=trace_id,
                created_at_ms=created_at_ms,
                sequence_index=sequence_index,
                usage=usage,
            )
        )
        state.last_usage[sequence_index] = usage

    finish_reason = _normalize_finish_reason(complete.get("finish_reason"))
    done_payload: dict[str, Any] = {
        "finish_reason": finish_reason,
        "sequence_index": sequence_index,
    }
    if output_ids:
        done_payload["token_ids"] = output_ids

    events.append(
        _base_event(
            "done",
            request_id,
            state,
            trace_id=trace_id,
            created_at_ms=created_at_ms,
            text_delta=_text_delta_for_position(
                complete,
                0,
                allow_full_text=True,
            )
            if not token_ids_to_emit
            else None,
            **done_payload,
        )
    )
    return events


def _base_event(
    kind: str,
    request_id: str,
    state: WorkerEventAdapterState,
    *,
    trace_id: str | None,
    created_at_ms: int | Callable[[], int] | None,
    **payload: Any,
) -> dict[str, Any]:
    event = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": MESSAGE_TYPE,
        "request_id": request_id,
        "kind": kind,
        "created_at_ms": _event_time_ms(created_at_ms, state.event_count),
    }
    state.event_count += 1
    if trace_id:
        event["trace_id"] = trace_id
    for key, value in payload.items():
        if value is not None:
            event[key] = value
    return event


def _event_time_ms(
    created_at_ms: int | Callable[[], int] | None,
    event_count: int,
) -> int:
    if created_at_ms is None:
        return int(time.time() * 1000)
    if callable(created_at_ms):
        return int(created_at_ms())
    return int(created_at_ms) + event_count


def _has_payload(response: dict[str, Any], key: str) -> bool:
    value = response.get(key)
    return isinstance(value, dict) and bool(value)


def _has_usage(payload: dict[str, Any]) -> bool:
    return any(
        key in payload
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens")
    )


def _build_usage(payload: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = _coerce_int(payload.get("prompt_tokens"), default=0)
    completion_tokens = _coerce_int(payload.get("completion_tokens"), default=0)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    cached_tokens = _coerce_int(payload.get("cached_tokens"), default=0)
    if cached_tokens:
        usage["cached_tokens"] = cached_tokens
    return usage


def _cached_tokens(usage: dict[str, int]) -> dict[str, int]:
    if usage.get("cached_tokens"):
        return {"cached_tokens": usage["cached_tokens"]}
    return {}


def _terminal_token_delta(output_ids: list[int], emitted_count: int) -> list[int]:
    if not output_ids:
        return []
    if emitted_count == 0:
        return output_ids
    if len(output_ids) > emitted_count:
        return output_ids[emitted_count:]
    return output_ids


def _map_output_logprobs(
    output_logprobs: dict[str, Any],
    position: int,
) -> dict[str, Any] | None:
    token_logprobs = output_logprobs.get("token_logprobs") or []
    top_logprobs = output_logprobs.get("top_logprobs") or []

    event_logprobs: dict[str, Any] = {}
    token_logprob = _list_get(token_logprobs, position)
    if token_logprob is not None:
        event_logprobs["token_logprob"] = float(token_logprob)

    top_entry = _list_get(top_logprobs, position)
    mapped_top = _map_top_logprobs(top_entry)
    if mapped_top:
        event_logprobs["top_logprobs"] = mapped_top

    return event_logprobs or None


def _text_delta_for_position(
    payload: dict[str, Any],
    position: int,
    *,
    allow_full_text: bool = True,
) -> str | None:
    for key in ("text_deltas", "texts", "output_texts"):
        text_delta = _coerce_text(_list_get(payload.get(key), position))
        if text_delta is not None:
            return text_delta

    if position == 0:
        for key in ("text_delta", "delta_text"):
            text_delta = _coerce_text(payload.get(key))
            if text_delta is not None:
                return text_delta

        if allow_full_text:
            for key in ("text", "output_text"):
                text_delta = _coerce_text(payload.get(key))
                if text_delta is not None:
                    return text_delta

    return None


def _map_top_logprobs(top_entry: Any) -> list[dict[str, Any]]:
    if not isinstance(top_entry, dict):
        return []

    values = top_entry.get("values") or []
    token_ids = top_entry.get("token_ids") or []
    mapped = []
    for idx, value in enumerate(values):
        if value is None:
            continue
        token_logprob: dict[str, Any] = {"logprob": float(value)}
        token_id = _list_get(token_ids, idx)
        if token_id is not None:
            token_logprob["token_id"] = _coerce_int(token_id)
        mapped.append(token_logprob)
    return mapped


def _map_error(error: dict[str, Any]) -> dict[str, Any]:
    status_code = str(error.get("http_status_code") or "").strip()
    code = f"sglang_http_{status_code}" if status_code else "sglang_error"
    details = {}
    if status_code:
        details["http_status_code"] = status_code
    if error.get("details"):
        details["details"] = error["details"]
    return {
        "code": code,
        "message": str(error.get("message") or "SGLang Generate failed"),
        "retryable": status_code in _RETRYABLE_HTTP_CODES,
        **({"details": details} if details else {}),
    }


def _normalize_finish_reason(raw: Any) -> str:
    if isinstance(raw, dict):
        raw = raw.get("type")
    value = str(raw or "stop")
    if value in _ALLOWED_FINISH_REASONS:
        return value
    return "error"


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _list_get(value: Any, index: int) -> Any:
    if not isinstance(value, list) or index < 0 or index >= len(value):
        return None
    return value[index]
