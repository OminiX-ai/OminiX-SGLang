# DeepSeek V4 Flash Final OminiX/SGLang Path

Date: 2026-05-02

This document records the accepted CUDA path for serving
`deepseek-ai/DeepSeek-V4-Flash` through OminiX-API and the headless
`sglang-ominix` scheduler boundary.

This is for DeepSeek V4 Flash. It is not a DeepSeek V4 Pro acceptance claim.

## Accepted Serving Chain

```text
client
  -> OminiX-API router-only
  -> OMINIX_V0_SCHEDULER_URL
  -> sglang-ominix v0 HTTP/SSE shim --mode grpc
  -> SglangScheduler.Generate
  -> DeepSeek-V4-Flash CUDA runtime, TP=8
  -> OminiX WorkerEvent SSE
  -> OpenAI-compatible JSON/SSE response from OminiX-API
```

OminiX-API owns public HTTP/OpenAI compatibility. This fork owns the
scheduler/runtime boundary below OminiX-API. The shim is intentionally not a
public OpenAI server.

## Accepted Live Report

The final c1+c4 live acceptance passed on the 8x RTX 5090 CUDA host:

```text
/root/autodl-tmp/ominix-cuda-dsv4flash/reports/ominix-sglang-dsv4flash-final-20260502-131114
```

The immediately preceding c1-only final-path run also passed:

```text
/root/autodl-tmp/ominix-cuda-dsv4flash/reports/ominix-sglang-dsv4flash-final-20260502-130843
```

## Hardware And Model

- Hardware: 8 x NVIDIA GeForce RTX 5090, 32 GB each.
- Tensor parallel size: `8`.
- Model path on the CUDA host:
  `/root/autodl-tmp/ominix-cuda-dsv4flash/models/deepseek-ai/DeepSeek-V4-Flash`.
- Model size on disk at verification time: about 149 GB.
- Data disk free space at verification time: about 1.7 TB under
  `/root/autodl-tmp`.

## Runtime Gates

The accepted run required these DeepSeek/SM120 CUDA gates:

```text
SGLANG_OPT_SM120_FLASHMLA_BACKEND=custom
OMINIX_CUDA_SM120_IMPL=extension
OMINIX_DSV4_MOE_GRAPH_SAFE_ROUTING=1
SGLANG_OPT_USE_TILELANG_MHC_PRE=1
SGLANG_OPT_USE_TILELANG_MHC_POST=1
SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
SGLANG_OPT_USE_TILELANG_INDEXER=1
SGLANG_TOPK_TRANSFORM_512_TORCH=0
SGLANG_OPT_USE_TOPK_V2=0
SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD=0
SGLANG_OPT_FIX_APE_2604=0
```

`OMINIX_DSV4_MOE_GRAPH_SAFE_ROUTING=1` is mandatory until graph-safe MoE
routing becomes the default code path. Without it, cold startup can fail during
CUDA graph capture.

## Launch Shape

The accepted harness launches three processes.

1. SGLang scheduler gRPC server:

```bash
PYTHONPATH=/root/autodl-tmp/ominix-cuda-dsv4flash/src/sglang-ominix/python \
/root/autodl-tmp/ominix-cuda-dsv4flash/venvs/sglang-cu130/bin/python \
  -m sglang.launch_server \
  --grpc-mode \
  --model-path /root/autodl-tmp/ominix-cuda-dsv4flash/models/deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer-path /root/autodl-tmp/ominix-cuda-dsv4flash/models/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30000 \
  --tp-size 8 \
  --pp-size 1 \
  --context-length 2048 \
  --mem-fraction-static 0.82 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 16384 \
  --max-running-requests 2 \
  --skip-server-warmup \
  --disable-flashinfer-autotune \
  --sampling-backend pytorch \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton_kernel \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend compressed \
  --reasoning-parser deepseek-v4
```

2. `sglang-ominix` v0 shim:

```bash
/root/autodl-tmp/ominix-cuda-dsv4flash/venvs/sglang-cu130/bin/python \
  scripts/ominix/v0_http_scheduler_shim.py \
  --mode grpc \
  --grpc-target 127.0.0.1:30000 \
  --host 127.0.0.1 \
  --port 19091 \
  --tokenizer-path /root/autodl-tmp/ominix-cuda-dsv4flash/models/deepseek-ai/DeepSeek-V4-Flash \
  --model-id deepseek-ai/DeepSeek-V4-Flash \
  --token '<configured scheduler token>'
```

3. OminiX-API router-only:

```bash
OMINIX_V0_SCHEDULER_URL=http://127.0.0.1:19091 \
OMINIX_V0_SCHEDULER_TOKEN='<configured scheduler token>' \
OMINIX_V0_SCHEDULER_TIMEOUT_SECS=1800 \
PORT=8080 \
/root/autodl-tmp/ominix-cuda-dsv4flash/build/ominix-api-router-only-target/debug/ominix-api-router-only
```

## Acceptance Gates

The final harness only claims acceptance after all of these pass:

- `GET /health`
- `GET /server_info`
- `GET /health_generate`
- `GET /get_model_info`
- `GET /get_loads`
- non-stream `/v1/chat/completions`
- streaming `/v1/chat/completions`, requiring multiple SSE chunks and `[DONE]`
- non-stream `/v1/completions`
- benchmark p128/o64/c1 and p256/o512/c1
- optional benchmark p256/o512/c4
- SM120 FlashMLA selected count at least the tensor-parallel size
- FlashMLA fallback count `0`
- no NaN/overflow log samples
- no CUDA graph capture errors
- launched processes stopped
- acceptance ports released
- GPUs idle after cleanup

## Final Result

From report
`/root/autodl-tmp/ominix-cuda-dsv4flash/reports/ominix-sglang-dsv4flash-final-20260502-131114`:

- Overall status: pass.
- Functional: pass.
- Benchmark: pass.
- SM120 FlashMLA selected count: `8`.
- FlashMLA fallback count: `0`.
- NaN/overflow count: `0`.
- CUDA graph error count: `0`.
- Cleanup: pass.
- Single-session decode: `39.95-42.18` output tok/s on the acceptance prompts.
- c4 per-request decode: `39.68-40.03` output tok/s.

The c4 elapsed times are serialized by the current server cap
`--max-running-requests 2`, but each accepted request keeps roughly the same
decode rate as the c1 path.

## Repo Components

This fork contributes the scheduler-side boundary:

- `scripts/ominix/v0_http_scheduler_shim.py`
  - fake mode for deterministic local tests
  - gRPC mode for `SglangScheduler.Generate`
  - `/get_model_info`, `/get_loads`, `/abort_request`, and `/flush_cache`
  - bearer-token enforcement
  - text-delta injection for token-id-only SGLang chunks
- `scripts/ominix/worker_token_boundary.py`
  - OminiX v0 token/text/completion/chat request preparation
  - tokenizer-backed text encoding
  - DeepSeek role-token fallback when `tokenizer.chat_template` is unset
  - stop token and stop string streaming handling
- `scripts/ominix/worker_event_adapter.py`
  - SGLang `GenerateResponse` dict stream to OminiX `WorkerEvent` stream
- `scripts/ominix/test_*.py`
  - local fake-mode, gRPC-stub, token-boundary, event-adapter, and
    cross-repo OminiX-API bridge coverage

## Verified Local Tests

From the fork root:

```bash
python3 -m py_compile scripts/ominix/*.py
python3 scripts/ominix/test_worker_token_boundary.py
python3 scripts/ominix/test_v0_http_scheduler_shim.py
python3 scripts/ominix/test_worker_event_adapter.py
python3 scripts/ominix/test_ominix_api_to_v0_shim.py
```

Latest local results:

- token boundary: 8 tests passed.
- v0 shim: 14 tests passed.
- worker event adapter: 5 tests passed.
- OminiX-API to v0 shim bridge: 2 tests passed.

## Remaining Production Hardening

The DeepSeek V4 Flash inference path is accepted at live integration level.
Remaining work is production hardening:

- package the shim/control process as a supervised service instead of a scripted
  harness launch.
- replace shim-local `/get_loads` fallback with a non-blocking native SGLang
  load API once available.
- wire HTTP disconnect-to-scheduler abort through a cancellable scheduler handle.
- generate typed Python/Rust bindings for the OminiX worker v0 protocol and
  replace dict-shaped local fixtures.
- continue SM120 fused FlashMLA/MHC kernel hardening and keep selected-vs-
  fallback log gates.
- build the direct OminiX-CUDA compute-plane adapter below SGLang's scheduler.

