#!/usr/bin/env python3
"""OminiX worker v0 tokenization and detokenization boundary helpers.

This module is dependency-light on purpose. The production shim should inject
the real SGLang/HuggingFace tokenizer object created for the launched model.
Tests can inject small stubs with compatible ``encode``, ``decode``, and
``apply_chat_template`` methods.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


PROTOCOL_VERSION = "ominix.worker.v0"
MESSAGE_TYPE_GENERATE_REQUEST = "GenerateRequest"
MESSAGE_TYPE_TOKENIZED_GENERATE = "TokenizedGenerate"
_DEEPSEEK_BOS_TOKEN = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
_DEEPSEEK_EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"
_DEEPSEEK_USER_TOKEN = "<\uff5cUser\uff5c>"
_DEEPSEEK_ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"


class TokenBoundaryError(ValueError):
    """Base class for request/token boundary validation errors."""


class MissingTokenizerError(TokenBoundaryError):
    """Raised when a text/chat request needs a tokenizer that was not injected."""


@dataclass(frozen=True)
class PreparedGenerate:
    request_id: str
    original_text: str
    input_ids: list[int]
    sampling_params: dict[str, Any]
    stream: bool


def prepare_generate_request(
    request: dict[str, Any],
    *,
    tokenizer: Any | None = None,
    chat_template: Any | None = None,
) -> PreparedGenerate:
    """Prepare OminiX worker v0 GenerateRequest JSON for scheduler Generate.

    The returned shape is the handoff OSO-013 can map to the current SGLang
    gRPC request: ``input_ids`` goes to tokenized input and ``sampling_params``
    goes to SGLang sampling fields.
    """

    if not isinstance(request, dict):
        raise TokenBoundaryError("GenerateRequest must be a JSON object")

    _validate_envelope(request)
    request_id = str(request.get("request_id") or f"ominix-{uuid.uuid4().hex}")
    sampling_params = _normalize_sampling_params(request)
    stream = bool(request.get("stream", False))

    message_type = request.get("message_type")
    if message_type == MESSAGE_TYPE_TOKENIZED_GENERATE:
        input_ids = _coerce_token_ids(request.get("input_ids"), "input_ids")
        original_text = _optional_text(request.get("original_text"))
        return PreparedGenerate(
            request_id=request_id,
            original_text=original_text,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
        )

    input_payload = request.get("input")
    if isinstance(input_payload, dict):
        kind = str(input_payload.get("kind") or "").strip().lower()
    else:
        input_payload = {}
        kind = ""

    if kind == "tokens" or "input_ids" in request:
        raw_tokens = input_payload.get("tokens", request.get("input_ids"))
        input_ids = _coerce_token_ids(raw_tokens, "input.tokens")
        original_text = _optional_text(input_payload.get("original_text"))
        return PreparedGenerate(
            request_id=request_id,
            original_text=original_text,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
        )

    if kind in {"text", "completion"} or _has_native_text(request):
        prompt = _extract_text_prompt(input_payload, request)
        input_ids = _encode_text(prompt, tokenizer, kind or "text")
        return PreparedGenerate(
            request_id=request_id,
            original_text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
        )

    if kind == "chat":
        messages = input_payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise TokenBoundaryError("input.kind='chat' requires non-empty messages")
        prompt_text, input_ids = _apply_chat_template(
            messages,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=input_payload.get("chat_template_kwargs"),
        )
        return PreparedGenerate(
            request_id=request_id,
            original_text=prompt_text,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
        )

    raise TokenBoundaryError(
        "GenerateRequest input.kind must be one of tokens, text, completion, or chat"
    )


class TokenDeltaDecoder:
    """Incrementally decode generated token IDs to text deltas.

    Stop strings are held back until it is clear they are not part of the output.
    This makes streaming and non-streaming accumulation match for simple text
    stop handling without importing SGLang's detokenizer process.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        stop_strings: Sequence[str] | None = None,
        stop_token_ids: Sequence[int] | None = None,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> None:
        if tokenizer is None:
            raise MissingTokenizerError("TokenDeltaDecoder requires a tokenizer")
        self.tokenizer = tokenizer
        self.stop_strings = tuple(text for text in (stop_strings or ()) if text)
        self.stop_token_ids = {int(token_id) for token_id in (stop_token_ids or ())}
        self.skip_special_tokens = bool(skip_special_tokens)
        self.spaces_between_special_tokens = bool(spaces_between_special_tokens)
        self._token_ids: list[int] = []
        self._sent_text = ""
        self._finished = False
        self._holdback_chars = (
            max((len(stop) for stop in self.stop_strings), default=0) - 1
        )
        if self._holdback_chars < 0:
            self._holdback_chars = 0

    def accept(self, token_ids: Sequence[int]) -> list[str]:
        if self._finished:
            return []

        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id in self.stop_token_ids:
                self._finished = True
                break
            self._token_ids.append(token_id)

        output_text, stopped_by_string = self._visible_output_text()
        if stopped_by_string:
            self._finished = True

        if self._finished:
            emit_until = len(output_text)
        else:
            emit_until = max(0, len(output_text) - self._holdback_chars)

        if emit_until <= len(self._sent_text):
            return []

        delta = output_text[len(self._sent_text) : emit_until]
        self._sent_text = output_text[:emit_until]
        return [delta] if delta else []

    def finish(self) -> str | None:
        output_text, _ = self._visible_output_text()
        if len(output_text) <= len(self._sent_text):
            self._finished = True
            return None

        delta = output_text[len(self._sent_text) :]
        self._sent_text = output_text
        self._finished = True
        return delta or None

    def _visible_output_text(self) -> tuple[str, bool]:
        text = _decode_token_ids(
            self.tokenizer,
            self._token_ids,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )
        stop_pos = _first_stop_position(text, self.stop_strings)
        if stop_pos is None:
            return text, False
        return text[:stop_pos], True


def _validate_envelope(request: dict[str, Any]) -> None:
    protocol_version = request.get("protocol_version")
    if protocol_version is not None and protocol_version != PROTOCOL_VERSION:
        raise TokenBoundaryError(
            f"unsupported protocol_version {protocol_version!r}; "
            f"expected {PROTOCOL_VERSION!r}"
        )

    message_type = request.get("message_type")
    if message_type is not None and message_type not in {
        MESSAGE_TYPE_GENERATE_REQUEST,
        MESSAGE_TYPE_TOKENIZED_GENERATE,
    }:
        raise TokenBoundaryError(
            f"unsupported message_type {message_type!r}; expected "
            f"{MESSAGE_TYPE_GENERATE_REQUEST!r} or {MESSAGE_TYPE_TOKENIZED_GENERATE!r}"
        )


def _normalize_sampling_params(request: dict[str, Any]) -> dict[str, Any]:
    raw_sampling = request.get("sampling")
    if raw_sampling is None:
        raw_sampling = request.get("sampling_params")
    if raw_sampling is None:
        sampling: dict[str, Any] = {}
    elif isinstance(raw_sampling, dict):
        sampling = dict(raw_sampling)
    else:
        raise TokenBoundaryError("sampling must be a JSON object when provided")

    if "max_new_tokens" not in sampling:
        for source_key in ("max_completion_tokens", "max_tokens"):
            if source_key in sampling:
                sampling["max_new_tokens"] = sampling[source_key]
                break

    return sampling


def _coerce_token_ids(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TokenBoundaryError(f"{field_name} must be a non-empty integer array")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise TokenBoundaryError(f"{field_name} must contain only integers") from exc


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _has_native_text(request: dict[str, Any]) -> bool:
    return isinstance(request.get("prompt"), str) or isinstance(request.get("text"), str)


def _extract_text_prompt(
    input_payload: dict[str, Any],
    request: dict[str, Any],
) -> str:
    for source in (input_payload, request):
        for key in ("prompt", "text"):
            value = source.get(key)
            if isinstance(value, str):
                return value
    raise TokenBoundaryError("text/completion input requires prompt or text")


def _encode_text(text: str, tokenizer: Any | None, input_kind: str) -> list[int]:
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise MissingTokenizerError(
            f"input.kind={input_kind!r} requires an injected tokenizer with encode(text)"
        )
    try:
        input_ids = tokenizer.encode(text)
    except TypeError:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
    return _coerce_token_ids(list(input_ids), "encoded input_ids")


def _apply_chat_template(
    messages: list[Any],
    *,
    tokenizer: Any | None,
    chat_template: Any | None,
    chat_template_kwargs: Any,
) -> tuple[str, list[int]]:
    if tokenizer is None:
        raise MissingTokenizerError(
            "input.kind='chat' requires an injected tokenizer with the model chat template"
        )

    kwargs = dict(chat_template_kwargs) if isinstance(chat_template_kwargs, dict) else {}
    template_fn = _chat_template_callable(tokenizer, chat_template)

    if template_fn is not None:
        try:
            result = template_fn(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **kwargs,
            )
        except Exception as exc:
            if not _is_missing_chat_template_error(exc):
                raise
        else:
            if isinstance(result, str):
                return result, _encode_text(result, tokenizer, "chat")

            input_ids = _coerce_token_ids(list(result), "chat template input_ids")
            prompt_text = _decode_token_ids(
                tokenizer,
                input_ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
            return prompt_text, input_ids

    prompt_text = _format_fallback_chat_prompt(messages, tokenizer)
    return prompt_text, _encode_text_no_special(prompt_text, tokenizer, "chat")


def _is_missing_chat_template_error(exc: Exception) -> bool:
    message = str(exc)
    return "chat_template" in message and "not set" in message


def _format_fallback_chat_prompt(messages: list[Any], tokenizer: Any) -> str:
    if _supports_deepseek_chat_tokens(tokenizer):
        return _format_deepseek_fallback_chat_prompt(messages, tokenizer)
    return _format_plain_fallback_chat_prompt(messages)


def _format_deepseek_fallback_chat_prompt(messages: list[Any], tokenizer: Any) -> str:
    parts: list[str] = []
    if _tokenizer_has_token(tokenizer, _DEEPSEEK_BOS_TOKEN):
        parts.append(_DEEPSEEK_BOS_TOKEN)

    pending_system: list[str] = []
    for raw_message in messages:
        role, content = _coerce_chat_message(raw_message)
        if role == "system":
            if content:
                pending_system.append(content)
            continue

        if pending_system:
            parts.append("System: " + "\n".join(pending_system).strip() + "\n")
            pending_system.clear()

        if role == "assistant":
            parts.append(f"{_DEEPSEEK_ASSISTANT_TOKEN}{content}")
            if content and _tokenizer_has_token(tokenizer, _DEEPSEEK_EOS_TOKEN):
                parts.append(_DEEPSEEK_EOS_TOKEN)
        elif role == "user":
            parts.append(f"{_DEEPSEEK_USER_TOKEN}{content}")
        else:
            parts.append(f"{role.title()}: {content}\n")

    if pending_system:
        parts.append("System: " + "\n".join(pending_system).strip() + "\n")
    if not parts or parts[-1] != _DEEPSEEK_ASSISTANT_TOKEN:
        parts.append(_DEEPSEEK_ASSISTANT_TOKEN)
    return "".join(parts)


def _format_plain_fallback_chat_prompt(messages: list[Any]) -> str:
    lines: list[str] = []
    for raw_message in messages:
        role, content = _coerce_chat_message(raw_message)
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }.get(role, role.title() or "User")
        lines.append(f"{label}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _coerce_chat_message(message: Any) -> tuple[str, str]:
    if not isinstance(message, dict):
        raise TokenBoundaryError("chat messages must be JSON objects")
    role = str(message.get("role") or "user").strip().lower()
    content = _chat_content_to_text(message.get("content"))
    return role, content


def _chat_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    pieces.append(text)
                elif isinstance(item.get("content"), str):
                    pieces.append(str(item["content"]))
        return "\n".join(piece for piece in pieces if piece)
    return str(content)


def _supports_deepseek_chat_tokens(tokenizer: Any) -> bool:
    return _tokenizer_has_token(tokenizer, _DEEPSEEK_USER_TOKEN) and _tokenizer_has_token(
        tokenizer,
        _DEEPSEEK_ASSISTANT_TOKEN,
    )


def _tokenizer_has_token(tokenizer: Any, token: str) -> bool:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            return token in get_vocab()
        except Exception:
            pass

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            token_id = convert(token)
        except Exception:
            return False
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        return token_id is not None and token_id != unk_token_id

    return False


def _encode_text_no_special(text: str, tokenizer: Any | None, input_kind: str) -> list[int]:
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise MissingTokenizerError(
            f"input.kind={input_kind!r} requires an injected tokenizer with encode(text)"
        )
    try:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        input_ids = tokenizer.encode(text)
    return _coerce_token_ids(list(input_ids), "encoded input_ids")


def _chat_template_callable(
    tokenizer: Any,
    chat_template: Any | None,
) -> Callable[..., Any] | None:
    if chat_template is not None:
        if hasattr(chat_template, "apply_chat_template"):
            return chat_template.apply_chat_template
        if callable(chat_template):
            return chat_template
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template
    return None


def _decode_token_ids(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens,
            )
        )
    except TypeError:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
            )
        )


def _first_stop_position(text: str, stop_strings: Sequence[str]) -> int | None:
    positions = [text.find(stop) for stop in stop_strings if stop and stop in text]
    if not positions:
        return None
    return min(positions)
