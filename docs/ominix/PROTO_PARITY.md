# OminiX v0 Proto Parity Bridge

The canonical protocol is
[`docs/strategy/ominix-worker-protocol-v0.md`](../../../../docs/strategy/ominix-worker-protocol-v0.md).
This note is only a bridge from the current SGLang scheduler gRPC smoke shape to
that OminiX v0 contract.

The current smoke client in `scripts/ominix/scheduler_smoke_client.py` still
uses the SGLang gRPC/proto shape:

```text
SglangScheduler.Generate(sglang.grpc.scheduler.GenerateRequest)
```

It must evolve toward the canonical OminiX v0 split:

- `GenerateRequest`: normalized API-to-scheduler request, above tokenization.
- `TokenizedGenerate`: trusted smoke/internal request, already tokenized.

Public OpenAI-compatible JSON stops at OminiX-API. Tokenization, sampling,
detokenization, structured-output state, request lifecycle, batching, and KV
policy remain owned by `sglang-ominix`.

## Current SGLang Smoke Source

The live Python server imports generated descriptors from the external
`smg_grpc_proto` package:

- Python modules:
  - `smg_grpc_proto.sglang_scheduler_pb2`
  - `smg_grpc_proto.sglang_scheduler_pb2_grpc`
- Current descriptor package expected by SGLang:
  `sglang.grpc.scheduler`
- Current service expected by SGLang:
  `sglang.grpc.scheduler.SglangScheduler`
- Current full methods:
  - `/sglang.grpc.scheduler.SglangScheduler/Generate`
  - `/sglang.grpc.scheduler.SglangScheduler/Abort`
  - `/sglang.grpc.scheduler.SglangScheduler/HealthCheck`
  - `/sglang.grpc.scheduler.SglangScheduler/GetModelInfo`
  - `/sglang.grpc.scheduler.SglangScheduler/GetLoads`

There is no source `*.proto` in this checkout. The checked-in Go generated
binding under `sgl-model-gateway/bindings/golang/internal/proto/` is useful as a
historical reference, but it is not current enough for the Python server.

## Canonical Target

For production traffic, OminiX v0 uses canonical `GenerateRequest` from the
strategy doc. It is normalized but not tokenized:

```text
protocol_version: "ominix.worker.v0"
message_type: "GenerateRequest"
request_id: string
trace_id: string
model: string
input: GenerateInput
sampling: SamplingParams
stream: boolean
```

For trusted smoke tests and internal scheduler validation, OminiX v0 uses
canonical `TokenizedGenerate`:

```text
protocol_version: "ominix.worker.v0"
message_type: "TokenizedGenerate"
request_id: string
trace_id: string
model: string
input_ids: integer[]
sampling: SamplingParams
stream: boolean
```

The current SGLang smoke client maps only to this second path conceptually. It
does not yet emit canonical OminiX envelopes or WorkerEvents.

## Tokenized Smoke Field Mapping

| Current SGLang field | Canonical OminiX v0 target | Required for smoke | Notes |
| --- | --- | --- | --- |
| `GenerateRequest.request_id` | `TokenizedGenerate.request_id` | yes | Stable request key used again by `AbortRequest.request_id`. |
| none | `protocol_version` | yes | Current SGLang proto has no protocol envelope. Target value is `ominix.worker.v0`. |
| none | `message_type` | yes | Target value is `TokenizedGenerate`. |
| none | `trace_id` | yes | Required by canonical v0; current smoke client does not send it. |
| none | `model` | yes | Required by canonical v0; current smoke relies on the already-started server model. |
| `GenerateRequest.tokenized.input_ids` | `TokenizedGenerate.input_ids` | yes | Direct tokenized ingress for trusted smoke/internal callers only. |
| `GenerateRequest.tokenized.original_text` | none | no | Debug/reference text only in current SGLang shape; not canonical v0. |
| none | `TokenizedGenerate.input_positions` | no | Optional canonical v0 field; current smoke does not send it. |
| none | `TokenizedGenerate.tokenizer` | no | Optional canonical v0 tokenizer identity/hash check; current smoke does not send it. |
| `GenerateRequest.sampling_params.max_new_tokens` | `sampling.max_completion_tokens` | yes | Must preserve field presence while using current SGLang proto because `grpc_server.py` calls `HasField("max_new_tokens")`. |
| `sampling_params.temperature` | `sampling.temperature` | yes | Smoke sets `0.0` for greedy behavior. |
| `sampling_params.top_p` | `sampling.top_p` | yes | Must be explicit in current proto. Proto3 `0.0` is invalid for SGLang semantics. |
| `sampling_params.top_k` | `sampling.top_k` | yes | Smoke sets `-1` so SGLang normalizes to all tokens, unless temperature forces greedy `top_k=1`. |
| `sampling_params.min_p` | `sampling.min_p` | no | Explicit default should be `0.0`. |
| `sampling_params.frequency_penalty` | `sampling.frequency_penalty` | no | Explicit default should be `0.0`. |
| `sampling_params.presence_penalty` | `sampling.presence_penalty` | no | Explicit default should be `0.0`. |
| `sampling_params.repetition_penalty` | `sampling.repetition_penalty` | yes | Must be explicit in current proto. Proto3 `0.0` does not match SGLang default `1.0`. |
| `sampling_params.n` | `sampling.n` | yes | Smoke uses `1`; non-streaming `n>1` has batch response behavior in current SGLang. |
| `sampling_params.stop` | `sampling.stop` | no | Optional stop strings. |
| `sampling_params.stop_token_ids` | `sampling.stop_token_ids` | no | Optional token stop IDs. |
| `sampling_params.min_new_tokens` | `sampling.min_tokens` | no | Name changes in canonical v0. |
| `sampling_params.regex` | `sampling.guided.regex` | no | One of canonical guided output options. |
| `sampling_params.json_schema` | `sampling.guided.json_schema` | no | Canonical type is object; current SGLang proto field is string. |
| `sampling_params.ebnf_grammar` | `sampling.guided.grammar` | no | Name changes in canonical v0. |
| `sampling_params.structural_tag` | no direct field | no | Needs an explicit v0 extension decision if required. |
| `sampling_params.logit_bias` | `sampling.logit_bias` | no | String-keyed token ID map in both shapes. |
| `sampling_params.stream_interval` | no direct field | no | Current server requires proto presence; canonical v0 should carry stream behavior via stream transport/options. |
| `sampling_params.custom_params` | `extra_body` or namespaced extension | no | Canonical v0 requires validation; silent loss is not allowed. |
| `sampling_params.skip_special_tokens` | response policy extension | no | Not in canonical `TokenizedGenerate`; normalized `GenerateRequest` has `response_options`. |
| `sampling_params.spaces_between_special_tokens` | response policy extension | no | Not in canonical `TokenizedGenerate`; requires an extension decision if needed. |
| `GenerateRequest.stream` | `TokenizedGenerate.stream` | yes | Smoke expects streaming token chunks. |
| `GenerateRequest.return_logprob` | `sampling.logprobs` and `stream_options.include_logprobs` | no | Current bool should map into canonical sampling/stream options. |
| `GenerateRequest.logprob_start_len` | `sampling.prompt_logprobs` | no | Semantics are not identical; requires adapter logic. |
| `GenerateRequest.top_logprobs_num` | `sampling.top_logprobs` | no | Optional top-k logprob count. |
| `GenerateRequest.token_ids_logprob` | no direct field | no | Needs an extension decision if required. |
| `GenerateRequest.lora_id` | backend/adapter extension | no | Not in canonical `TokenizedGenerate`; likely belongs under a validated backend extension. |
| `GenerateRequest.disaggregated_params.*` | `kv_reuse_hint`/backend extension | no | Canonical v0 handles KV/Mooncake hints in normalized `GenerateRequest`; current bootstrap fields are SGLang-specific. |

Fields present in the current generated Go binding but not wired by
`grpc_server.py::_convert_generate_request` are not part of v0 smoke parity:
`mm_inputs`, `return_hidden_states`, `custom_logit_processor`, `timestamp`,
`log_metrics`, `input_embeds`, and `data_parallel_rank`.

## Stream Response Mapping

The current SGLang server streams `GenerateResponse` messages. Canonical OminiX
v0 streams semantic `WorkerEvent` records that OminiX-API can convert to public
SSE deltas.

The next adapter design is in
[`WORKER_EVENT_ADAPTER_PLAN.md`](WORKER_EVENT_ADAPTER_PLAN.md), with a
dependency-free dict transformer in
[`scripts/ominix/worker_event_adapter.py`](../../scripts/ominix/worker_event_adapter.py).

| Current SGLang field | Canonical OminiX v0 target | Notes |
| --- | --- | --- |
| `GenerateResponse.request_id` | `WorkerEvent.request_id` | Mirrors request ID. |
| none | `WorkerEvent.protocol_version` | Target value is `ominix.worker.v0`. |
| none | `WorkerEvent.message_type` | Target value is `WorkerEvent`. |
| none | `WorkerEvent.trace_id` | Required by canonical tracing model when available. |
| `GenerateResponse.chunk.token_ids` | `WorkerEvent.kind="token"`, `token_ids` | Incremental generated token IDs. |
| `GenerateResponse.chunk.prompt_tokens` | `WorkerEvent.usage.prompt_tokens` | Current field is cumulative count. |
| `GenerateResponse.chunk.completion_tokens` | `WorkerEvent.usage.completion_tokens` | Current field is cumulative count. |
| `GenerateResponse.chunk.cached_tokens` | `WorkerEvent.usage.cached_tokens` | Cache hit count. |
| `GenerateResponse.chunk.output_logprobs` | `WorkerEvent.logprobs` | Adapter must map current output-logprob arrays to canonical `LogprobInfo`. |
| `GenerateResponse.chunk.input_logprobs` | `WorkerEvent.logprobs` or prompt-logprob extension | Usually only first chunk; canonical mapping needs adapter logic. |
| `GenerateResponse.chunk.index` | `WorkerEvent.sequence_index` | Needed for `n>1` ordering. |
| `GenerateResponse.complete.output_ids` | `WorkerEvent.kind="done"`, `token_ids` | Final generated token IDs. |
| `GenerateResponse.complete.finish_reason` | `WorkerEvent.finish_reason` | Current values include `stop`, `length`, and `abort`. |
| `GenerateResponse.complete.matched_token_id` | stop-detail extension | Optional stop match detail; no direct canonical field. |
| `GenerateResponse.complete.matched_stop_str` | stop-detail extension | Optional stop match detail; no direct canonical field. |
| `GenerateResponse.error.message` | `WorkerEvent.kind="error"`, `error.message` | Error text from the Python server. |
| `GenerateResponse.error.http_status_code` | `WorkerEvent.error.code` or extension | Current field is stringly typed. Canonical v0 uses `ErrorInfo`. |
| `GenerateResponse.error.details` | `WorkerEvent.error.details` | Optional traceback/details. |

## Control RPC Mapping

| Current SGLang RPC | Canonical OminiX v0 target | Required for smoke | Notes |
| --- | --- | --- | --- |
| `Abort(AbortRequest)` | `AbortRequest` | yes | Canonical request adds `protocol_version`, `message_type`, `trace_id`, and reason enum/string. |
| `HealthCheck(HealthCheckRequest)` | `HealthCheck` -> `HealthStatus` | yes | Current Python server submits an internal one-token request; canonical health must be cheap and avoid model-memory allocation. |
| `GetModelInfo(GetModelInfoRequest)` | `GetModelInfo` -> `ModelInfo` | yes | Current Python response includes model/tokenizer/config metadata but not the full canonical identity/capability shape. |
| `GetLoads(GetLoadsRequest)` | `GetLoads` -> `CapacityReport` | yes | Current request uses `include`/`dp_rank`; canonical request uses `include_ranks`/`include_kv`. |

## Exact Blockers

1. Source proto blocker: no `*.proto` file exists in this checkout, including the
   brief-path `sgl-model-gateway/src/proto/sglang_scheduler.proto`.
2. Package/envelope blocker: the current SGLang package/service is
   `sglang.grpc.scheduler.SglangScheduler`; no generated OminiX v0 proto/package
   exists in this checkout for the canonical protocol.
3. Python package blocker: runtime parity depends on external
   `smg_grpc_proto>=0.3.3`. The local system Python environment does not have
   `grpc` or `smg_grpc_proto` installed.
4. Generated binding blocker: the checked-in Go binding does not expose
   `GetLoads`, `GetLoadsRequest`, `GetLoadsResponse`, `SchedulerLoad`,
   `AggregateMetrics`, or the load metric messages used by
   `grpc_server.py`.
5. `GetModelInfo` parity blocker: the Python server constructs response fields
   `architectures`, `id2label_json`, and `num_labels`; the checked-in Go binding
   does not include those fields.
6. Presence blocker: `SamplingParams.max_new_tokens`,
   `SamplingParams.stream_interval`, and `GetLoadsRequest.dp_rank` must retain
   proto presence or the Python server conversion code must stop using
   `HasField(...)`.
7. Default-value blocker: proto3 zero values do not match SGLang sampling
   defaults. Internal OminiX callers must send explicit sampling defaults, or
   `sglang-ominix` must fill SGLang defaults before constructing
   `SGLSamplingParams`.
8. Canonical envelope blocker: current smoke requests do not carry
   `protocol_version`, `message_type`, `trace_id`, or `model`, all required by
   canonical OminiX v0 `TokenizedGenerate`.
9. Stream event blocker: current SGLang `GenerateResponse` is not canonical
   `WorkerEvent`; response adaptation is required before OminiX-API can treat
   the stream as v0 protocol output.
10. Health semantic blocker: current SGLang `HealthCheck` exercises generation;
    canonical OminiX v0 health must report liveness/readiness cheaply without
    allocating model work.

## Local Checker

Validate the installed Python gRPC/proto package for the current SGLang smoke
shape without starting a model server:

```bash
python3 scripts/ominix/scheduler_smoke_client.py --check-proto-only
```

The checker validates imports, descriptor package, service methods, request and
response message fields, presence-sensitive fields, and construction of the
minimal SGLang tokenized smoke `GenerateRequest`. It does not validate the
canonical OminiX v0 envelope yet because that generated proto/package does not
exist in this checkout.
