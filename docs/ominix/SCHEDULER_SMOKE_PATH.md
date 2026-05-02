# OminiX Scheduler gRPC Smoke Path

This runbook is for the milestone-1 `sglang-ominix` scheduler-only path. It
does not use `http_server.py`, OpenAI HTTP routes, Ollama routes, Anthropic
routes, or the Rust gateway HTTP facade. OminiX-API owns public HTTP/OpenAI
traffic; this fork owns the scheduler/runtime lifecycle below that boundary.

## Current Status

Canonical protocol reference:
[`docs/strategy/ominix-worker-protocol-v0.md`](../../../../docs/strategy/ominix-worker-protocol-v0.md).
This runbook still uses the current SGLang gRPC/proto shape for smoke testing;
it does not yet emit canonical OminiX v0 `GenerateRequest`,
`TokenizedGenerate`, or `WorkerEvent` envelopes. The current-to-v0 field bridge
and blockers are tracked in [`PROTO_PARITY.md`](PROTO_PARITY.md).

The Python scheduler-only server path exists:

- `python/sglang/launch_server.py` selects gRPC when `--grpc-mode` is set.
- `python/sglang/srt/entrypoints/grpc_server.py` starts a standalone gRPC
  server, launches scheduler processes with
  `launch_scheduler_process_only(...)`, and registers
  `sglang.grpc.scheduler.SglangScheduler`.
- `python/sglang/srt/grpc/grpc_request_manager.py` accepts already-tokenized
  requests and forwards `TokenizedGenerateReqInput`, `AbortReq`, and
  `GetLoadsReqInput` to the scheduler over ZMQ.
- `python/sglang/srt/grpc/scheduler_launcher.py` launches scheduler process
  rank(s) without tokenizer/detokenizer processes.

The repo currently has proto drift:

- The brief path `sgl-model-gateway/src/proto/sglang_scheduler.proto` is not
  present in this checkout.
- No `*.proto` files are present under `vendor_research/sglang-ominix`.
- The checked-in generated Go binding at
  `sgl-model-gateway/bindings/golang/internal/proto/sglang_scheduler_grpc.pb.go`
  exposes `Generate`, `Embed`, `HealthCheck`, `Abort`, `GetModelInfo`, and
  `GetServerInfo`, but not `GetLoads`.
- `grpc_server.py` implements `GetLoads`, so the installed
  `smg-grpc-proto` package must be checked for parity before this smoke path is
  considered complete.

## Required Dependencies

Use Python 3.10 or newer. The system `python3` checked on this machine is 3.9
and is not sufficient for the package metadata.

The gRPC scheduler path needs the normal SGLang runtime deps plus these
gRPC-specific deps declared in `python/pyproject.toml`:

- `smg-grpc-proto>=0.3.3`
- `grpcio>=1.78.0`
- `grpcio-reflection>=1.78.0`
- `grpcio-health-checking>=1.78.0`
- `pyzmq>=25.1.2`
- `numpy`, `torch`, `transformers-kt`, and model runtime deps

Recommended setup from the fork root:

```bash
cd /Users/cloud/home/CannFusion/vendor_research/sglang-ominix
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./python
```

If the full package install is too heavy for client-only probing, install at
least:

```bash
python -m pip install 'grpcio>=1.78.0' 'grpcio-health-checking>=1.78.0' \
  'grpcio-reflection>=1.78.0' 'smg-grpc-proto>=0.3.3'
```

Local blocker observed before installing anything:

```text
python3 with PYTHONPATH=python:
  sglang: FAIL ModuleNotFoundError: No module named 'numpy'
  grpc: FAIL ModuleNotFoundError: No module named 'grpc'
  grpc_health.v1: FAIL ModuleNotFoundError: No module named 'grpc_health'
  grpc_reflection.v1alpha: FAIL ModuleNotFoundError: No module named 'grpc_reflection'
  smg_grpc_proto: FAIL ModuleNotFoundError: No module named 'smg_grpc_proto'
```

## Launch Command

Start one gRPC scheduler server. This command uses dummy weights to avoid
downloading model weights, but it still needs the model config and a CUDA-capable
runtime.

```bash
cd /Users/cloud/home/CannFusion/vendor_research/sglang-ominix
source .venv/bin/activate

PYTHONPATH=python python -m sglang.launch_server \
  --grpc-mode \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --load-format dummy \
  --host 127.0.0.1 \
  --port 30000 \
  --tp-size 1 \
  --skip-server-warmup \
  --mem-fraction-static 0.30 \
  --max-total-tokens 4096
```

For a real output smoke, remove `--load-format dummy` and use a tiny real model
that fits the test GPU, for example `Qwen/Qwen2.5-0.5B` or
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

Do not use:

```bash
python -m sglang.launch_server --model-path ...   # no --grpc-mode, starts HTTP
python -m sglang_router.launch_server ...         # starts router/facade path
```

## Ports

Expected external listener:

- `127.0.0.1:30000` by default, or `--host/--port` from the launch command.
- This is gRPC, not HTTP.

Expected internal scheduler IPC with default `enable_dp_attention=false`:

- ZMQ uses temporary `ipc://...` endpoints for scheduler input, detokenizer
  output, tokenizer IPC placeholders, RPC IPC, and metrics IPC.
- NCCL uses a free port unless `--nccl-port` is supplied.

If `--enable-dp-attention` is used, `PortArgs.init_new(...)` switches internal
ZMQ to TCP. With `--port 30000`, expected derived ports are:

- `30233`: dist init (`port + 233`)
- `30234`: tokenizer IPC placeholder
- `30235`: detokenizer output IPC
- `30236`: scheduler RPC IPC
- `30237`: metrics IPC
- `30238`: scheduler input IPC
- `--nccl-port`: supplied or dynamically selected

If `--disaggregation-mode prefill` is used, `grpc_server.py` also starts the
disaggregation bootstrap service on `--disaggregation-bootstrap-port`, default
`8998`.

## Smoke Client

Check the installed Python gRPC/proto package without starting a model server:

```bash
cd /Users/cloud/home/CannFusion/vendor_research/sglang-ominix
source .venv/bin/activate

python scripts/ominix/scheduler_smoke_client.py --check-proto-only
```

Run the checked-in tokenized smoke client after the server is listening:

```bash
cd /Users/cloud/home/CannFusion/vendor_research/sglang-ominix
source .venv/bin/activate

python scripts/ominix/scheduler_smoke_client.py \
  --target 127.0.0.1:30000 \
  --input-ids 1,2,3 \
  --max-new-tokens 8
```

Abort path:

```bash
python scripts/ominix/scheduler_smoke_client.py \
  --target 127.0.0.1:30000 \
  --input-ids 1,2,3 \
  --max-new-tokens 128 \
  --abort-after-first-chunk
```

The client calls, in order:

1. `HealthCheck(HealthCheckRequest)`
2. `GetModelInfo(GetModelInfoRequest)`
3. `GetLoads(GetLoadsRequest(include=["all"]))` if the installed proto exposes it
4. `Generate(GenerateRequest)` with `TokenizedInput.input_ids`
5. `Abort(AbortRequest)` when `--abort-after-first-chunk` is set

These are SGLang gRPC method and message names. They are compatibility smoke
steps, not the canonical OminiX v0 protocol surface.

Expected `Generate` behavior:

- The request must contain `tokenized.input_ids`.
- `stream=true` returns `GenerateResponse.chunk.token_ids` until final
  `GenerateResponse.complete.output_ids`, or an error response.
- The gRPC server converts the request into `TokenizedGenerateReqInput` with
  `tokenizer=None` normalization. This bypasses public HTTP tokenization.
- The response returns token IDs. The smoke client does not detokenize.

Expected `Abort` behavior:

- The client sends `AbortRequest.request_id`.
- `GrpcRequestManager.abort_request(...)` sends `AbortReq(rid=request_id)` to
  the scheduler and marks local request state finished.
- A stream may finish before the abort arrives if the smoke request is too
  short; use a larger `--max-new-tokens` for the abort smoke.

## OminiX v0 HTTP Shim

`scripts/ominix/v0_http_scheduler_shim.py` is the current executable boundary
between OminiX-API and this scheduler-only fork. It is not a public OpenAI API
server. It accepts OminiX worker v0 `GenerateRequest` JSON on `/generate` and
returns `WorkerEvent` records as SSE:

```bash
cd /Users/cloud/home/CannFusion/vendor_research/sglang-ominix

python3 scripts/ominix/v0_http_scheduler_shim.py \
  --host 127.0.0.1 \
  --port 19091 \
  --token bridge-check-token
```

Wire OminiX-API to the shim:

```bash
cd /Users/cloud/home/CannFusion/vendor_research/OminiX-API

OMINIX_V0_SCHEDULER_URL=http://127.0.0.1:19091 \
OMINIX_V0_SCHEDULER_TOKEN=bridge-check-token \
OMINIX_V0_SCHEDULER_TIMEOUT_SECS=5 \
./target/debug/ominix-api-router-only --host 127.0.0.1 --port 18080
```

Current shim mode:

- `--mode fake` emits deterministic text `pong`.
- `--mode grpc --grpc-target host:port` bridges tokenized OminiX v0 requests to
  `SglangScheduler.Generate`.
- The emitted fake SGLang-like `GenerateResponse` dicts flow through
  `scripts/ominix/worker_event_adapter.py`, so token, usage, finish, trace, and
  text-delta behavior uses the same adapter intended for real gRPC responses.
- The shim validates optional `protocol_version` and `message_type` fields and
  accepts both typed OminiX v0 requests and legacy native `/generate` request
  shapes during the transition.
- Bearer token validation is enabled when `--token` is supplied.

Run the gRPC-backed shim after the scheduler server is listening:

```bash
python3 scripts/ominix/v0_http_scheduler_shim.py \
  --mode grpc \
  --grpc-target 127.0.0.1:30000 \
  --tokenizer-path /path/to/deepseek-ai/DeepSeek-V4-Flash \
  --model-id deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 \
  --port 19091 \
  --token bridge-check-token
```

First accepted gRPC input slice:

- `input.kind="tokens"` with `input.tokens: integer[]`
- transitional native `input_ids: integer[]`
- `input.kind="text"`, `input.kind="completion"`, and `input.kind="chat"` when
  `--tokenizer-path` is supplied and `transformers.AutoTokenizer` can load the
  model tokenizer.

Text, chat, completion, and responses inputs return HTTP 400 when they require
tokenization and no tokenizer has been configured.

Control routes exposed by the shim:

- `GET /get_model_info`
- `GET /get_loads`
- `POST /abort_request`
- `POST /flush_cache`

Verification:

```bash
python3 scripts/ominix/test_worker_event_adapter.py
python3 scripts/ominix/test_v0_http_scheduler_shim.py
python3 scripts/ominix/test_ominix_api_to_v0_shim.py
```

This shim closes the first OminiX-API to `sglang-ominix` executable contract.
Live CUDA execution against the real `SglangScheduler.Generate` endpoint and
DeepSeek-V4-Flash weights is accepted in
[`DEEPSEEK_V4_FLASH_FINAL_PATH.md`](DEEPSEEK_V4_FLASH_FINAL_PATH.md).

Expected `HealthCheck` behavior:

- Custom scheduler `HealthCheck` submits a one-token internal request with
  `input_ids=[0]`.
- Standard gRPC health is also registered for service
  `sglang.grpc.scheduler.SglangScheduler`.

Expected `GetModelInfo` behavior:

- Returns model path, tokenizer path, generation flag, served model name,
  context length, vocab size, architecture metadata, and token IDs from
  `scheduler_info` plus `ModelConfig`.

Expected `GetLoads` behavior:

- `grpc_server.py` maps the request to `GrpcRequestManager.get_loads(...)`.
- The manager sends `GetLoadsReqInput` to the scheduler and waits for
  `GetLoadsReqOutput`.
- This call is blocked until proto parity is confirmed, because the checked-in
  Go binding lacks `GetLoads`.

## Scheduler Assumptions

Tokenizer bypass:

- gRPC `GenerateRequest` must have `tokenized.input_ids`.
- `_convert_generate_request(...)` creates `TokenizedGenerateReqInput` directly.
- `sampling_params.normalize(tokenizer=None)` is used.

Detokenizer bypass:

- No detokenizer process is launched.
- `GrpcRequestManager` binds the scheduler output endpoint named
  `detokenizer_ipc_name` and consumes `BatchTokenIDOutput` directly.
- Responses are token ID chunks/completions for OminiX-API to map to SSE later.

Request structs used:

- `TokenizedGenerateReqInput`
- `TokenizedEmbeddingReqInput` for deferred embedding path
- `AbortReq`
- `GetLoadsReqInput` and `GetLoadsReqOutput`
- `HealthCheckOutput`
- `BatchTokenIDOutput`

Still-public API types:

- The Python CLI still labels `--host` and `--port` under the "HTTP server"
  argument group even when `--grpc-mode` is set.
- `sglang/launch_server.py` still defaults to HTTP when `--grpc-mode` is not
  supplied.
- `sgl-model-gateway` still owns public OpenAI facade routes in this fork copy;
  do not route milestone-1 smoke traffic through it.

## Blockers To Clear

1. Install/runtime blocker: this checkout was not in a working Python env.
   `python3` is 3.9 and missing `numpy`, `grpc`, `grpc_health`,
   `grpc_reflection`, and `smg_grpc_proto`.
2. Proto source blocker: `sgl-model-gateway/src/proto/sglang_scheduler.proto`
   does not exist in the checkout, and no `*.proto` file exists under
   `vendor_research/sglang-ominix`.
3. Proto parity blocker: checked-in generated Go bindings do not expose
   `GetLoads`, while `grpc_server.py` implements it. The final CUDA path uses
   Python native SGLang proto modules successfully, but `GetLoads` can still
   block at runtime; the shim returns local load metadata when the unary RPC
   times out.
4. Model/runtime blocker: a real smoke requires a GPU/runtime compatible with
   this SGLang fork. `--load-format dummy` reduces weight download cost but
   still exercises model config, scheduler startup, kernels, and memory setup.
5. Warmup token blocker: `grpc_server.py` warmup uses arbitrary token IDs.
   `--skip-server-warmup` avoids that warmup request; the smoke client then
   controls input IDs explicitly.
