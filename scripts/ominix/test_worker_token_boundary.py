#!/usr/bin/env python3
"""Fixture tests for OminiX worker v0 token boundary helpers."""

from __future__ import annotations

import unittest

from worker_token_boundary import (
    MissingTokenizerError,
    TokenDeltaDecoder,
    prepare_generate_request,
)


class StubTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "ping": 11,
            "hello": 12,
            "system:You": 21,
            "are": 22,
            "concise.": 23,
            "user:Say": 24,
            "pong.": 25,
            "assistant:": 26,
        }
        self.inverse_vocab = {token_id: text for text, token_id in self.vocab.items()}
        self.inverse_vocab.update(
            {
                101: "po",
                102: "ng",
                201: "hello",
                202: " ST",
                203: "OP",
                301: "kept",
                999: "<eos>",
            }
        )

    def encode(self, text: str) -> list[int]:
        return [self.vocab[word] for word in text.split()]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        del spaces_between_special_tokens
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == 999:
                continue
            pieces.append(self.inverse_vocab[token_id])
        return "".join(pieces)


class StubChatTemplate:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        **_: object,
    ):
        self.last_args = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": return_dict,
        }
        text = " ".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            text = f"{text} assistant:"
        return text


class MissingTemplateDeepSeekTokenizer:
    def __init__(self) -> None:
        self.last_encode_kwargs = None

    def get_vocab(self):
        return {
            "<\uff5cbegin\u2581of\u2581sentence\uff5c>": 0,
            "<\uff5cend\u2581of\u2581sentence\uff5c>": 1,
            "<\uff5cUser\uff5c>": 2,
            "<\uff5cAssistant\uff5c>": 3,
        }

    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        raise ValueError(
            "Cannot use chat template functions because tokenizer.chat_template is not set"
        )

    def encode(self, text: str, **kwargs) -> list[int]:
        self.last_encode_kwargs = kwargs
        return [max(2, ord(char) % 32000) for char in text]


class WorkerTokenBoundaryTest(unittest.TestCase):
    def test_token_input_bypasses_tokenizer(self):
        prepared = prepare_generate_request(
            {
                "protocol_version": "ominix.worker.v0",
                "message_type": "GenerateRequest",
                "request_id": "req-token",
                "input": {
                    "kind": "tokens",
                    "tokens": [7, "8", 9],
                    "original_text": "already tokenized",
                },
                "sampling": {"max_completion_tokens": 4},
                "stream": True,
            }
        )

        self.assertEqual(prepared.request_id, "req-token")
        self.assertEqual(prepared.input_ids, [7, 8, 9])
        self.assertEqual(prepared.original_text, "already tokenized")
        self.assertEqual(prepared.sampling_params["max_new_tokens"], 4)
        self.assertTrue(prepared.stream)

    def test_completion_text_uses_injected_tokenizer(self):
        prepared = prepare_generate_request(
            {
                "request_id": "req-text",
                "input": {"kind": "completion", "prompt": "ping hello"},
                "sampling_params": {"max_tokens": 2, "temperature": 0.0},
            },
            tokenizer=StubTokenizer(),
        )

        self.assertEqual(prepared.original_text, "ping hello")
        self.assertEqual(prepared.input_ids, [11, 12])
        self.assertEqual(prepared.sampling_params["max_new_tokens"], 2)
        self.assertEqual(prepared.sampling_params["temperature"], 0.0)

    def test_chat_uses_stub_template_and_tokenizer(self):
        template = StubChatTemplate()
        prepared = prepare_generate_request(
            {
                "request_id": "req-chat",
                "input": {
                    "kind": "chat",
                    "messages": [
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Say pong."},
                    ],
                },
            },
            tokenizer=StubTokenizer(),
            chat_template=template,
        )

        self.assertEqual(
            prepared.original_text,
            "system:You are concise. user:Say pong. assistant:",
        )
        self.assertEqual(prepared.input_ids, [21, 22, 23, 24, 25, 26])
        self.assertEqual(
            template.last_args,
            {"tokenize": True, "add_generation_prompt": True, "return_dict": False},
        )

    def test_chat_falls_back_to_deepseek_role_tokens_without_template(self):
        tokenizer = MissingTemplateDeepSeekTokenizer()
        prepared = prepare_generate_request(
            {
                "request_id": "req-chat-fallback",
                "input": {
                    "kind": "chat",
                    "messages": [
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Say pong."},
                    ],
                },
            },
            tokenizer=tokenizer,
        )

        self.assertIn("<\uff5cbegin\u2581of\u2581sentence\uff5c>", prepared.original_text)
        self.assertIn("<\uff5cUser\uff5c>Say pong.", prepared.original_text)
        self.assertTrue(prepared.original_text.endswith("<\uff5cAssistant\uff5c>"))
        self.assertGreater(len(prepared.input_ids), 0)
        self.assertEqual(tokenizer.last_encode_kwargs, {"add_special_tokens": False})

    def test_chat_without_tokenizer_has_clear_dependency_error(self):
        with self.assertRaisesRegex(MissingTokenizerError, "requires an injected tokenizer"):
            prepare_generate_request(
                {
                    "request_id": "req-chat-missing",
                    "input": {
                        "kind": "chat",
                        "messages": [{"role": "user", "content": "Say pong."}],
                    },
                }
            )

    def test_incremental_detokenization(self):
        decoder = TokenDeltaDecoder(StubTokenizer())

        self.assertEqual(decoder.accept([101]), ["po"])
        self.assertEqual(decoder.accept([102]), ["ng"])
        self.assertIsNone(decoder.finish())

    def test_stop_string_truncates_final_text(self):
        decoder = TokenDeltaDecoder(StubTokenizer(), stop_strings=[" STOP"])

        self.assertEqual(decoder.accept([201]), ["h"])
        self.assertEqual(decoder.accept([202]), ["ell"])
        self.assertEqual(decoder.accept([203]), ["o"])
        self.assertIsNone(decoder.finish())

    def test_stop_token_id_is_not_decoded(self):
        decoder = TokenDeltaDecoder(StubTokenizer(), stop_token_ids=[999])

        self.assertEqual(decoder.accept([301, 999]), ["kept"])
        self.assertIsNone(decoder.finish())


if __name__ == "__main__":
    unittest.main()
