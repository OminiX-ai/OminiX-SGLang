# SGLang GenerateResponse To OminiX WorkerEvent Adapter Plan

This is the next adapter step for the milestone-1 scheduler smoke path. The
gRPC server remains unchanged. The adapter consumes dict-shaped current SGLang
`GenerateResponse` messages and emits dict-shaped canonical OminiX
`WorkerEvent` messages.

Canonical target:
[`docs/strategy/ominix-worker-protocol-v0.md`](../../../../docs/strategy/ominix-worker-protocol-v0.md)

Prototype helper:
[`scripts/ominix/worker_event_adapter.py`](../../scripts/ominix/worker_event_adapter.py)

Fixture:
[`scripts/ominix/test_worker_event_adapter.py`](../../scripts/ominix/test_worker_event_adapter.py)

## Boundary

Current runtime path:

```text
SGLang gRPC GenerateResponse stream
  -> MessageToDict(..., preserving_proto_field_name=True)
  -> pure-Python WorkerEvent adapter
  -> OminiX WorkerEvent dict stream
```

This is a compatibility adapter. It does not import `grpc`, `protobuf`, SGLang,
or generated proto modules. It can run in a plain Python environment and can be
used by smoke clients or future API-side tests before an OminiX generated proto
exists in this checkout.

Do not modify these server files for this step:

- `python/sglang/srt/entrypoints/grpc_server.py`
- `python/sglang/srt/grpc/grpc_request_manager.py`
- generated `smg_grpc_proto` modules

## Input Shape

The adapter accepts the current response variants after protobuf-to-dict
conversion:

```python
{"request_id": "req-1", "chunk": {...}}
{"request_id": "req-1", "complete": {...}}
{"request_id": "req-1", "error": {...}}
```

Expected `chunk` fields:

- `token_ids`
- `prompt_tokens`
- `completion_tokens`
- `cached_tokens`
- `output_logprobs`
- `input_logprobs`
- `index`

Expected `complete` fields:

- `output_ids`
- `finish_reason`
- `prompt_tokens`
- `completion_tokens`
- `cached_tokens`
- `output_logprobs`
- `input_logprobs`
- `index`
- `matched_token_id`
- `matched_stop_str`

Expected `error` fields:

- `message`
- `http_status_code`
- `details`

## Output Shape

Every emitted event includes:

```text
protocol_version: "ominix.worker.v0"
message_type: "WorkerEvent"
request_id: string
trace_id: string optional
kind: WorkerEventKind
created_at_ms: integer
```

The adapter currently emits these event kinds:

- `prefill_done`
- `token`
- `usage`
- `error`
- `done`

The adapter does not emit `queued`, `scheduled`, `text`, or `kv_feedback`
because the current SGLang `GenerateResponse` stream does not carry that
information.

## Mapping

| Current SGLang dict path | WorkerEvent path | Notes |
| --- | --- | --- |
| `request_id` | `request_id` | Required. |
| external adapter argument | `trace_id` | Optional because current SGLang response has no trace field. |
| `chunk.prompt_tokens` | `prefill_done.usage.prompt_tokens` | Emitted once per `sequence_index` when prompt usage first appears. |
| `chunk.cached_tokens` | `prefill_done.usage.cached_tokens` | Only emitted when non-zero. |
| `chunk.token_ids[]` | `token.token_id` | One WorkerEvent per token ID so logprobs can stay position-aligned. |
| `chunk.output_logprobs.token_logprobs[]` | `token.logprobs.token_logprob` | Position-aligned with emitted token IDs. |
| `chunk.output_logprobs.top_logprobs[]` | `token.logprobs.top_logprobs[]` | Current SGLang top-logprob entries have token IDs but no decoded token text. |
| `chunk.index` | `sequence_index` | Needed when `n > 1`. |
| `complete.output_ids[]` | terminal `token` events and `done.token_ids` | See terminal-token rule below. |
| `complete.finish_reason` | `done.finish_reason` | Unknown values normalize to `error`. |
| `complete.prompt_tokens` / `completion_tokens` / `cached_tokens` | `usage.usage` | Emitted before `done` when usage fields are present. |
| `error.message` | `error.error.message` | Defaults to `SGLang Generate failed` if missing. |
| `error.http_status_code` | `error.error.code` | Current code format is `sglang_http_<status>`. |
| `error.details` | `error.error.details.details` | Wrapped because canonical `ErrorInfo.details` is an object. |

## Terminal-Token Rule

The current gRPC server can place final streaming token IDs in
`complete.output_ids` rather than a final `chunk`. For that reason the adapter
synthesizes terminal `token` events from `complete.output_ids`.

The helper tracks emitted token counts per `sequence_index`:

- If no prior token events were emitted for that sequence, every `output_id`
  becomes a token event. This covers non-streaming final-only responses.
- If `output_ids` is longer than the emitted count, only the suffix is emitted.
  This covers cumulative final output.
- Otherwise, all `output_ids` are treated as the terminal fragment. This covers
  the current streaming server shape where the final fragment can be shorter
  than the prior emitted count.

The `done` event still carries `token_ids` for audit/debug parity. OminiX-API
should build public SSE deltas only from `token` or future `text` events.

## Known Gaps

Text deltas:

- Current tokenized gRPC smoke responses do not detokenize.
- The adapter emits `token_id`, not `text_delta`.
- A future scheduler-owned detokenizer boundary should emit `text` events or
  add `text_delta` to token events before crossing into OminiX-API.

Prompt/input logprobs:

- Current SGLang can send `input_logprobs`, usually on the first chunk or final
  response.
- Canonical v0 has a token-level `LogprobInfo`, but prompt logprob placement is
  not final.
- The prototype intentionally maps only output logprobs. Add a canonical
  prompt-logprob extension before exposing `input_logprobs`.

Matched stops:

- Current `matched_token_id` and `matched_stop_str` have no direct canonical
  field.
- Keep them out of the WorkerEvent top level until a namespaced stop-detail
  extension is accepted.

Timing and queue lifecycle:

- Current `GenerateResponse` does not include queue, prefill, decode, or
  first-token timings.
- Future server-side integration should emit `queued`, `scheduled`,
  `prefill_done.timing`, and final `usage.timing` from scheduler lifecycle
  state, not infer them at the API edge.

Errors:

- HTTP-like status codes are a compatibility artifact from current SGLang gRPC.
- The next typed protocol should emit deterministic OminiX error codes directly.

## Integration Sequence

1. Keep the current SGLang gRPC server unchanged.
2. Use the pure-Python helper in smoke tests to validate event semantics from
   dict-shaped `GenerateResponse` streams.
3. Add an optional smoke-client flag later, for example
   `--print-worker-events`, that runs `MessageToDict` output through the helper.
4. Once OminiX v0 proto/structs exist, replace dict output with typed
   `WorkerEvent` construction at the scheduler/API boundary.
5. Move text-delta and prompt-logprob handling only after the canonical
   extensions are explicitly defined.

## Local Validation

```bash
python3 -m py_compile \
  scripts/ominix/worker_event_adapter.py \
  scripts/ominix/test_worker_event_adapter.py

python3 scripts/ominix/test_worker_event_adapter.py
```
