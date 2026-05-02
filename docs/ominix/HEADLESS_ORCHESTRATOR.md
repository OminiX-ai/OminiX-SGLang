# SGLang-OminiX Headless Orchestrator

Local fork workspace: `vendor_research/sglang-ominix`
Branch: `ominix-headless-orchestrator`
Seed: `vendor_research/sglang-kt`

## Purpose

`sglang-ominix` is a fork of SGLang that removes SGLang as the public API owner and keeps it as a scheduler/orchestrator runtime under OminiX.

OminiX-API owns:

- OpenAI-compatible HTTP routes
- auth, quota, admission, model routing
- streaming SSE formatting
- admin and observability endpoints

`sglang-ominix` owns:

- request lifecycle below API
- tokenizer and detokenizer ownership
- sampling, logprobs, penalties, and structured-output state
- continuous batching
- request abort/pause/resume
- KV bookkeeping and scheduling policy
- worker health/load routing
- normalized internal ingress plus tokenized gRPC smoke/test ingress

OminiX compute owns:

- CUDA/Ascend execution
- DeepSeek V4 Flash kernels
- tensor/expert parallel collectives
- Mooncake KV import/export

## Milestone 1 Boundary

Keep:

- `python/sglang/srt/grpc/scheduler_launcher.py`
- `python/sglang/srt/grpc/grpc_request_manager.py`
- `python/sglang/srt/entrypoints/grpc_server.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/tp_worker.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/mem_cache/`

Ignore or feature-disable:

- Python HTTP/OpenAI/Ollama/Anthropic public entrypoints
- public tokenizer/detokenizer HTTP route ownership
- Rust gateway OpenAI facade routes
- response persistence, MCP, WASM, external OpenAI worker mode

Do not insert below `TpModelWorker` for milestone 1. That would become a model-runtime fork, not a pure orchestrator fork.

## Minimum Smoke Path

```text
OminiX test client
  -> tokenized gRPC Generate
  -> grpc_server.py
  -> grpc_request_manager.py
  -> scheduler ZMQ loop
  -> current model runner
  -> streamed token ids
```

Concrete launch and client commands are in
[`SCHEDULER_SMOKE_PATH.md`](SCHEDULER_SMOKE_PATH.md).
Current SGLang gRPC smoke fields and their OminiX v0 bridge mapping are in
[`PROTO_PARITY.md`](PROTO_PARITY.md).

This smoke path proves the scheduler can run without the public SGLang HTTP server. It is not the final OminiX-API product boundary. The production boundary keeps tokenizer and sampler ownership in `sglang-ominix` so OminiX-API does not duplicate tokenization, penalties, logprobs, or structured-output behavior.

Required methods:

- `Generate`
- `Abort`
- `HealthCheck`
- `GetModelInfo`
- `GetLoads`

Deferred:

- public OpenAI route handling
- embeddings
- rerank/score/classify
- multimodal
- tool call formatting
- response persistence

## Worker Protocol Direction

The canonical protocol is
[`docs/strategy/ominix-worker-protocol-v0.md`](../../../../docs/strategy/ominix-worker-protocol-v0.md).
Start from the existing SGLang scheduler proto subset only for smoke/proto
validation, then evolve it toward the canonical OminiX-owned protocol before
production.

- `WorkerHello`
- `GenerateRequest`
- `TokenizedGenerate` for trusted smoke/internal validation only
- `AbortRequest`
- `HealthCheck`
- `GetLoads`
- `GetModelInfo`
- `WorkerEvent`

OpenAI JSON belongs above this fork in OminiX-API. Tokenizer and sampler ownership belongs inside `sglang-ominix`; direct tokenized ingress is reserved for smoke tests and trusted internal callers.

## Compute Runtime Direction

The later integration replaces SGLang model execution with an OminiX compute runtime behind the scheduler:

```text
prefill(PrefillBatch) -> StepResult
decode(DecodeBatch) -> StepResult
export_kv(...) -> KvLease
import_kv(...) -> KvImportResult
release_kv(...) -> void
get_capacity() -> CapacityReport
```

CUDA and Ascend device handles must stay below this boundary.
Sampling should stay above this boundary in `sglang-ominix`.
