#!/usr/bin/env python3
"""Fixture tests for the pure-Python WorkerEvent adapter."""

from __future__ import annotations

import unittest

from worker_event_adapter import adapt_generate_response_stream


class WorkerEventAdapterTest(unittest.TestCase):
    def test_streaming_chunk_and_terminal_fragment(self):
        responses = [
            {
                "request_id": "req-1",
                "chunk": {
                    "token_ids": [101],
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "cached_tokens": 2,
                    "output_logprobs": {
                        "token_logprobs": [-0.1],
                        "top_logprobs": [
                            {"values": [-0.1, -1.5], "token_ids": [101, 201]}
                        ],
                    },
                    "index": 0,
                },
            },
            {
                "request_id": "req-1",
                "complete": {
                    "output_ids": [102],
                    "finish_reason": "stop",
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "cached_tokens": 2,
                    "output_logprobs": {
                        "token_logprobs": [-0.2],
                        "top_logprobs": [
                            {"values": [-0.2, -2.0], "token_ids": [102, 202]}
                        ],
                    },
                    "index": 0,
                },
            },
        ]

        events = list(
            adapt_generate_response_stream(
                responses, trace_id="trace-1", created_at_ms=1000
            )
        )

        self.assertEqual(
            [event["kind"] for event in events],
            ["prefill_done", "token", "token", "usage", "done"],
        )
        self.assertEqual([event["created_at_ms"] for event in events], list(range(1000, 1005)))
        self.assertEqual(events[0]["usage"]["prompt_tokens"], 3)
        self.assertEqual(events[0]["usage"]["cached_tokens"], 2)
        self.assertEqual(events[1]["token_id"], 101)
        self.assertEqual(events[1]["logprobs"]["token_logprob"], -0.1)
        self.assertEqual(events[1]["logprobs"]["top_logprobs"][1]["token_id"], 201)
        self.assertEqual(events[2]["token_id"], 102)
        self.assertEqual(events[3]["usage"]["total_tokens"], 5)
        self.assertEqual(events[4]["finish_reason"], "stop")
        self.assertEqual(events[4]["token_ids"], [102])
        self.assertTrue(all(event["trace_id"] == "trace-1" for event in events))

    def test_non_streaming_complete_synthesizes_token_events(self):
        responses = [
            {
                "request_id": "req-2",
                "complete": {
                    "output_ids": [301, 302],
                    "finish_reason": "length",
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "index": 1,
                },
            }
        ]

        events = list(adapt_generate_response_stream(responses, created_at_ms=2000))

        self.assertEqual(
            [event["kind"] for event in events], ["token", "token", "usage", "done"]
        )
        self.assertEqual([event.get("sequence_index") for event in events], [1, 1, 1, 1])
        self.assertEqual([event.get("token_id") for event in events[:2]], [301, 302])
        self.assertEqual(events[-1]["finish_reason"], "length")

    def test_chunk_text_deltas_are_preserved(self):
        responses = [
            {
                "request_id": "req-text",
                "chunk": {
                    "token_ids": [401, 402],
                    "text_deltas": ["po", "ng"],
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                    "index": 0,
                },
            }
        ]

        events = list(adapt_generate_response_stream(responses, created_at_ms=4000))

        self.assertEqual(
            [event["kind"] for event in events], ["prefill_done", "token", "token"]
        )
        self.assertEqual([event.get("text_delta") for event in events[1:]], ["po", "ng"])

    def test_complete_text_is_preserved_when_no_chunks_were_emitted(self):
        responses = [
            {
                "request_id": "req-complete-text",
                "complete": {
                    "output_ids": [501],
                    "text": "pong",
                    "finish_reason": "stop",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "index": 0,
                },
            }
        ]

        events = list(adapt_generate_response_stream(responses, created_at_ms=5000))

        self.assertEqual(
            [event["kind"] for event in events], ["token", "usage", "done"]
        )
        self.assertEqual(events[0]["token_id"], 501)
        self.assertEqual(events[0]["text_delta"], "pong")

    def test_error_maps_to_canonical_error_info(self):
        responses = [
            {
                "request_id": "req-3",
                "error": {
                    "message": "scheduler unavailable",
                    "http_status_code": "503",
                    "details": "traceback omitted",
                },
            }
        ]

        events = list(adapt_generate_response_stream(responses, created_at_ms=3000))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "error")
        self.assertEqual(events[0]["finish_reason"], "error")
        self.assertEqual(events[0]["error"]["code"], "sglang_http_503")
        self.assertTrue(events[0]["error"]["retryable"])
        self.assertEqual(events[0]["error"]["details"]["http_status_code"], "503")


if __name__ == "__main__":
    unittest.main()
